// Render pipeline: pass A (opaque + ice) -> depth of field -> pass B (liquids)
// -> pass C (outer glass) -> bloom -> filmic grade. Liquids and glass refract an
// already defocused copy of the scene, so the background seen through the drink
// is blurred while the glass itself stays sharp.
import * as THREE from 'three';
import { FullScreenQuad } from 'three/addons/postprocessing/Pass.js';

const VERT = /* glsl */`
varying vec2 vUv;
void main() { vUv = uv; gl_Position = vec4( position.xy, 0.0, 1.0 ); }
`;

const LINZ = /* glsl */`
uniform float uNear, uFar;
float linZ( float d ) { float z = d * 2.0 - 1.0; return 2.0 * uNear * uFar / ( uFar + uNear - z * ( uFar - uNear ) ); }
`;

const shader = (fragmentShader, uniforms) => new THREE.ShaderMaterial({
	vertexShader: VERT, fragmentShader, uniforms, depthTest: false, depthWrite: false,
});

function rt(w, h, opts = {}) {
	return new THREE.WebGLRenderTarget(w, h, {
		type: THREE.HalfFloatType, minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter,
		depthBuffer: false, ...opts,
	});
}

export class Pipeline {
	constructor(renderer, W, H) {
		this.renderer = renderer;
		this.W = W; this.H = H;
		const depthTexture = new THREE.DepthTexture(W, H, THREE.UnsignedIntType);
		this.main = rt(W, H, { depthBuffer: true, depthTexture });
		const mip = { generateMipmaps: true, minFilter: THREE.LinearMipmapLinearFilter };
		this.sceneA = rt(W, H, mip);
		this.sceneB = rt(W, H, mip);
		this.half = rt(W >> 1, H >> 1);
		this.halfBlur = rt(W >> 1, H >> 1);
		this.bloom = [rt(W >> 2, H >> 2), rt(W >> 2, H >> 2), rt(W >> 3, H >> 3), rt(W >> 3, H >> 3), rt(W >> 4, H >> 4), rt(W >> 4, H >> 4)];
		this.quad = new FullScreenQuad();

		this.copyMat = shader(/* glsl */`
			uniform sampler2D tSrc; varying vec2 vUv;
			void main() { gl_FragColor = texture2D( tSrc, vUv ); }
		`, { tSrc: { value: null } });

		this.cocMat = shader(/* glsl */`
			uniform sampler2D tColor, tDepth; uniform vec2 uTexel; varying vec2 vUv;
			${LINZ}
			void main() {
				vec2 o = uTexel * 0.5;
				vec3 c = texture2D( tColor, vUv + vec2( -o.x, -o.y ) ).rgb + texture2D( tColor, vUv + vec2( o.x, -o.y ) ).rgb
					+ texture2D( tColor, vUv + vec2( -o.x, o.y ) ).rgb + texture2D( tColor, vUv + vec2( o.x, o.y ) ).rgb;
				gl_FragColor = vec4( c * 0.25, linZ( texture2D( tDepth, vUv ).x ) );
			}
		`, { tColor: { value: null }, tDepth: { value: depthTexture }, uTexel: { value: new THREE.Vector2(1 / W, 1 / H) }, uNear: { value: 1 }, uFar: { value: 1000 } });

		// single-pass gather bokeh (scatter-as-gather on a golden-angle spiral)
		this.blurMat = shader(/* glsl */`
			uniform sampler2D tHalf; uniform vec2 uTexel; uniform float uFocus, uAperture, uMaxBlur; varying vec2 vUv;
			float coc( float z ) { return min( uAperture * abs( 1.0 / uFocus - 1.0 / z ), uMaxBlur ); }
			void main() {
				vec4 c0 = texture2D( tHalf, vUv );
				float cz = c0.a, cs = coc( cz );
				vec3 acc = c0.rgb; float tot = 1.0;
				float radius = 0.6, ang = 0.0;
				for ( int i = 0; i < 90; i ++ ) {
					if ( radius >= uMaxBlur ) break;
					vec2 tc = vUv + vec2( cos( ang ), sin( ang ) ) * uTexel * radius;
					vec4 s = texture2D( tHalf, tc );
					float ss = coc( s.a );
					if ( s.a > cz ) ss = clamp( ss, 0.0, cs * 2.0 );
					float m = smoothstep( radius - 0.5, radius + 0.5, ss );
					acc += mix( acc / tot, s.rgb, m );
					tot += 1.0;
					radius += 1.1 / radius;
					ang += 2.39996323;
				}
				gl_FragColor = vec4( acc / tot, cs );
			}
		`, { tHalf: { value: this.half.texture }, uTexel: { value: new THREE.Vector2(2 / W, 2 / H) }, uFocus: { value: 30 }, uAperture: { value: 300 }, uMaxBlur: { value: 14 } });

		this.dofMat = shader(/* glsl */`
			uniform sampler2D tColor, tDepth, tBlur; uniform float uFocus, uAperture; varying vec2 vUv;
			${LINZ}
			void main() {
				vec3 sharp = texture2D( tColor, vUv ).rgb;
				float z = linZ( texture2D( tDepth, vUv ).x );
				float cs = uAperture * abs( 1.0 / uFocus - 1.0 / z );
				vec4 b = texture2D( tBlur, vUv );
				float w = smoothstep( 0.7, 2.6, cs );
				gl_FragColor = vec4( mix( sharp, b.rgb, w ), 1.0 );
			}
		`, { tColor: { value: null }, tDepth: { value: depthTexture }, tBlur: { value: this.halfBlur.texture }, uFocus: { value: 30 }, uAperture: { value: 600 }, uNear: { value: 1 }, uFar: { value: 1000 } });

		this.brightMat = shader(/* glsl */`
			uniform sampler2D tSrc; uniform float uThreshold; varying vec2 vUv;
			void main() {
				vec3 c = texture2D( tSrc, vUv ).rgb;
				float l = max( c.r, max( c.g, c.b ) );
				float k = smoothstep( uThreshold, uThreshold * 2.5, l );
				gl_FragColor = vec4( min( c * k, vec3( 30.0 ) ), 1.0 );
			}
		`, { tSrc: { value: null }, uThreshold: { value: 1.6 } });

		this.gaussMat = shader(/* glsl */`
			uniform sampler2D tSrc; uniform vec2 uDir; varying vec2 vUv;
			void main() {
				vec3 c = texture2D( tSrc, vUv ).rgb * 0.227027;
				c += texture2D( tSrc, vUv + uDir * 1.3846 ).rgb * 0.316216 + texture2D( tSrc, vUv - uDir * 1.3846 ).rgb * 0.316216;
				c += texture2D( tSrc, vUv + uDir * 3.2308 ).rgb * 0.070270 + texture2D( tSrc, vUv - uDir * 3.2308 ).rgb * 0.070270;
				gl_FragColor = vec4( c, 1.0 );
			}
		`, { tSrc: { value: null }, uDir: { value: new THREE.Vector2() } });

		this.finalMat = shader(/* glsl */`
			uniform sampler2D tColor, tB1, tB2, tB3; uniform vec2 uRes;
			uniform float uExposure, uBloom, uWhip, uWhipDir, uTime, uGrain, uVignette, uFade;
			varying vec2 vUv;
			vec3 aces( vec3 x ) {
				const mat3 m1 = mat3( 0.59719, 0.07600, 0.02840, 0.35458, 0.90834, 0.13383, 0.04823, 0.01566, 0.83777 );
				const mat3 m2 = mat3( 1.60475, -0.10208, -0.00327, -0.53108, 1.10813, -0.07276, -0.07367, -0.00605, 1.07602 );
				vec3 v = m1 * x;
				vec3 a = v * ( v + 0.0245786 ) - 0.000090537;
				vec3 b = v * ( 0.983729 * v + 0.4329510 ) + 0.238081;
				return clamp( m2 * ( a / b ), 0.0, 1.0 );
			}
			vec3 srgb( vec3 c ) { return mix( c * 12.92, 1.055 * pow( c, vec3( 1.0 / 2.4 ) ) - 0.055, step( 0.0031308, c ) ); }
			float hash( vec2 p ) { vec3 p3 = fract( vec3( p.xyx ) * 0.1031 ); p3 += dot( p3, p3.yzx + 33.33 ); return fract( ( p3.x + p3.y ) * p3.z ); }
			vec3 sceneAt( vec2 uv ) {
				vec3 c = texture2D( tColor, uv ).rgb;
				// mild lateral chromatic aberration towards the frame edges
				vec2 d = ( uv - 0.5 ) * 0.0016;
				c.r = texture2D( tColor, uv + d ).r;
				c.b = texture2D( tColor, uv - d ).b;
				vec3 bl = texture2D( tB1, uv ).rgb * 0.5 + texture2D( tB2, uv ).rgb * 0.3 + texture2D( tB3, uv ).rgb * 0.35;
				return c + bl * uBloom;
			}
			void main() {
				vec3 c;
				if ( uWhip > 0.001 ) {
					c = vec3( 0.0 );
					for ( int i = 0; i < 20; i ++ ) {
						float f = float( i ) / 19.0 - 0.5;
						c += sceneAt( vUv + vec2( uWhipDir * f * 0.16 * uWhip, f * 0.01 * uWhip ) );
					}
					c /= 20.0;
				} else {
					c = sceneAt( vUv );
				}
				c *= uExposure * vec3( 1.035, 1.0, 0.95 );
				c = aces( c );
				// gentle S-curve and warm shadows
				c = mix( c, c * c * ( 3.0 - 2.0 * c ), 0.18 );
				c += vec3( 0.012, 0.006, 0.0 ) * ( 1.0 - c );
				float vig = smoothstep( 1.15, 0.35, length( ( vUv - 0.5 ) * vec2( 1.0, 1.35 ) ) );
				c *= mix( 1.0, vig, uVignette );
				c = srgb( c );
				c += ( hash( vUv * uRes + fract( uTime * 13.17 ) * 917.0 ) - 0.5 ) * uGrain;
				c *= uFade;
				gl_FragColor = vec4( clamp( c, 0.0, 1.0 ), 1.0 );
			}
		`, {
			tColor: { value: this.main.texture }, tB1: { value: this.bloom[1].texture }, tB2: { value: this.bloom[3].texture }, tB3: { value: this.bloom[5].texture },
			uRes: { value: new THREE.Vector2(W, H) }, uExposure: { value: 1.0 }, uBloom: { value: 0.12 }, uWhip: { value: 0 }, uWhipDir: { value: 1 },
			uTime: { value: 0 }, uGrain: { value: 0.022 }, uVignette: { value: 0.32 }, uFade: { value: 1 },
		});
	}

	pass(material, target, uniforms = {}) {
		for (const k in uniforms) material.uniforms[k].value = uniforms[k];
		this.quad.material = material;
		this.renderer.setRenderTarget(target);
		this.quad.render(this.renderer);
	}

	// set.groups: { A: [], B: [], C: [] } of objects whose visibility is toggled per pass
	render(scene, camera, groups, look) {
		const r = this.renderer;
		const gl = r.getContext();
		const prof = window.PROFILE ? [] : null;
		let tp = performance.now();
		const px = new Float32Array(4), px8 = new Uint8Array(4);
		const mark = (name) => { if (!prof) return; if (r.getRenderTarget()) gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.FLOAT, px); else gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px8); const n = performance.now(); prof.push(`${name}:${(n - tp).toFixed(0)}`); tp = n; };
		// userData.active carries the choreography's own show/hide state
		const vis = (key) => { for (const k of ['A', 'B', 'C']) for (const o of groups[k]) o.visible = k === key && o.userData.active !== false; };
		const near = camera.near, far = camera.far;

		// --- pass A
		vis('A');
		r.shadowMap.needsUpdate = true;
		r.setRenderTarget(this.main);
		r.setClearColor(0x000000, 1);
		r.clear(true, true, true);
		r.render(scene, camera);
		mark('A');

		// --- depth of field on pass A
		const apHalf = look.aperture / 2;
		this.pass(this.cocMat, this.half, { tColor: this.main.texture, uNear: near, uFar: far });
		this.pass(this.blurMat, this.halfBlur, { uFocus: look.focusDist, uAperture: apHalf, uMaxBlur: 14 * this.W / 1080 });
		this.pass(this.dofMat, this.sceneA, { tColor: this.main.texture, uFocus: look.focusDist, uAperture: look.aperture, uNear: near, uFar: far });
		// write the defocused image back under the existing depth buffer
		this.copyMat.uniforms.tSrc.value = this.sceneA.texture;
		this.quad.material = this.copyMat;
		r.setRenderTarget(this.main);
		this.quad.render(r);
		mark('dof');

		// --- pass B: liquids refract sceneA
		vis('B');
		r.render(scene, camera);
		this.pass(this.copyMat, this.sceneB, { tSrc: this.main.texture });
		mark('B');

		// --- pass C: outer glass refracts sceneB
		vis('C');
		r.setRenderTarget(this.main);
		r.render(scene, camera);
		mark('C');

		// --- bloom
		const [b0, b1, b2, b3, b4, b5] = this.bloom;
		this.pass(this.brightMat, b0, { tSrc: this.main.texture });
		const blur = (src, tmp, dst, w, h) => {
			this.pass(this.gaussMat, tmp, { tSrc: src.texture, uDir: new THREE.Vector2(1 / w, 0) });
			this.pass(this.gaussMat, dst, { tSrc: tmp.texture, uDir: new THREE.Vector2(0, 1 / h) });
		};
		blur(b0, b1, b0, b0.width, b0.height);
		this.pass(this.copyMat, b1, { tSrc: b0.texture });
		this.pass(this.copyMat, b2, { tSrc: b0.texture });
		blur(b2, b3, b2, b2.width, b2.height);
		this.pass(this.copyMat, b3, { tSrc: b2.texture });
		this.pass(this.copyMat, b4, { tSrc: b2.texture });
		blur(b4, b5, b4, b4.width, b4.height);
		this.pass(this.copyMat, b5, { tSrc: b4.texture });

		// --- grade to the canvas
		this.pass(this.finalMat, null, {
			uWhip: look.whip, uWhipDir: look.whipDir, uTime: look.time, uExposure: look.exposure ?? 1, uFade: look.fade ?? 1,
		});
		mark('bloom+final');
		if (prof) console.log('profile ' + prof.join(' '));
	}
}
