// Materials with shader injections for the multi-pass refraction pipeline:
//   pass A: opaque set + ice + inner glass skins   (built-in transmission)
//   pass B: liquids, refracting a copy of pass A    (uSceneTex)
//   pass C: outer glass skins, refracting pass A+B  (uSceneTex)
import * as THREE from 'three';

export const GLSL_NOISE = /* glsl */`
float hash13( vec3 p3 ) {
	p3 = fract( p3 * 0.1031 );
	p3 += dot( p3, p3.zyx + 31.32 );
	return fract( ( p3.x + p3.y ) * p3.z );
}
float noise3( vec3 x ) {
	vec3 i = floor( x ); vec3 f = fract( x ); f = f * f * ( 3.0 - 2.0 * f );
	return mix( mix( mix( hash13( i ), hash13( i + vec3( 1, 0, 0 ) ), f.x ),
	                 mix( hash13( i + vec3( 0, 1, 0 ) ), hash13( i + vec3( 1, 1, 0 ) ), f.x ), f.y ),
	            mix( mix( hash13( i + vec3( 0, 0, 1 ) ), hash13( i + vec3( 1, 0, 1 ) ), f.x ),
	                 mix( hash13( i + vec3( 0, 1, 1 ) ), hash13( i + vec3( 1, 1, 1 ) ), f.x ), f.y ), f.z );
}
float fbm3( vec3 p ) { return noise3( p ) * 0.55 + noise3( p * 2.03 ) * 0.3 + noise3( p * 4.11 ) * 0.15; }
`;

const header = (s, extra) => s.replace('#include <common>', `#include <common>\n${extra}`);

// Chunks are only expanded after onBeforeCompile, so expand the transmission
// helpers here and point them at our own scene copy instead of the renderer's buffer.
const sceneTexPars = THREE.ShaderChunk.transmission_pars_fragment
	.replace('uniform sampler2D transmissionSamplerMap;', 'uniform sampler2D uSceneTex;')
	.replaceAll('transmissionSamplerMap', 'uSceneTex');

// ---------------------------------------------------------------- glass

export function glassOuterMaterial(sceneTex, { wall, base, normalMap = null, tint = new THREE.Color(0.95, 0.99, 0.97) }) {
	const m = new THREE.MeshPhysicalMaterial({
		color: 0xffffff, metalness: 0, roughness: 0.015, transmission: 1, ior: 1.5, thickness: wall,
		specularIntensity: 1, envMapIntensity: 1.15, attenuationColor: tint, attenuationDistance: 14,
		normalMap, normalScale: new THREE.Vector2(0, 0),
	});
	m.userData.u = { uSceneTex: sceneTex, uWall: { value: wall }, uBaseY: { value: base }, uFog: { value: 0 } };
	m.onBeforeCompile = (s) => {
		Object.assign(s.uniforms, m.userData.u);
		s.vertexShader = header(s.vertexShader, 'varying float vLocalY;')
			.replace('#include <begin_vertex>', '#include <begin_vertex>\nvLocalY = position.y;');
		s.fragmentShader = header(s.fragmentShader, 'varying float vLocalY;\nuniform float uWall;\nuniform float uBaseY;\nuniform float uFog;')
			.replace('#include <transmission_fragment>', THREE.ShaderChunk.transmission_fragment.replace(
				'material.thickness = thickness;',
				// glass looks thicker at grazing angles and through the solid base
				`float ndv = abs( dot( normalize( normal ), normalize( vViewPosition ) ) );
				material.thickness = ( vLocalY < uBaseY ? 2.4 : uWall ) / max( ndv, 0.16 );`) +
				// misted condensation scatters a little light back
				'\ntotalDiffuse = mix( totalDiffuse, vec3( 0.62, 0.64, 0.65 ), uFog * 0.22 );')
			.replace('#include <transmission_pars_fragment>', sceneTexPars);
	};
	return m;
}

export function glassInnerMaterial({ wall }) {
	return new THREE.MeshPhysicalMaterial({
		color: 0xffffff, metalness: 0, roughness: 0.02, transmission: 1, ior: 1.5, thickness: wall * 2,
		specularIntensity: 1, envMapIntensity: 1.0,
	});
}

// ---------------------------------------------------------------- liquid

// Coffee / water volume. Concentration field gives the layered espresso look,
// Beer-Lambert absorption along the refracted ray gives amber edges and dark core.
export function liquidMaterials(sceneTex, opts = {}) {
	const u = {
		uSceneTex: sceneTex,
		uTime: { value: 0 },
		uLevel: { value: -100 },
		uCenter: { value: new THREE.Vector3() },
		uRadius: { value: new THREE.Vector2(3.2, 0.05) }, // r(y) = x + y * localY
		uBottom: { value: 1.1 },
		uSimple: { value: 0 }, // 1 -> constant thickness (tilted / thin bodies)
		uThick: { value: 1 },
		uSigma: { value: new THREE.Vector3(0.4, 0.74, 1.15) },
		uWaterSigma: { value: new THREE.Vector3(0.022, 0.009, 0.005) },
		uConcBase: { value: 0 },
		uConcTop: { value: 0 },
		uLayer: { value: 0 },
		uLayerSoft: { value: 1.2 },
		uMix: { value: 0 },
		uSwirl: { value: 0 },
		uPlume: { value: new THREE.Vector4(0, 0, 0, 0) }, // x, z, strength, -
		uFoam: { value: 0 },
		uCrema: { value: 0 },
		uFoamColor: { value: new THREE.Color(0.55, 0.36, 0.2) },
		uRipple: { value: new THREE.Vector4(0, 0, 0, 0) }, // x, z, amp, -
		uTurb: { value: 0.02 },
		...opts.uniforms,
	};

	const decl = `
		${GLSL_NOISE}
		uniform float uTime, uLevel, uBottom, uSimple, uThick, uConcBase, uConcTop, uLayer, uLayerSoft, uMix, uSwirl, uFoam, uCrema, uTurb;
		uniform vec3 uCenter, uSigma, uWaterSigma, uFoamColor;
		uniform vec2 uRadius;
		uniform vec4 uPlume, uRipple;

		float coffeeConc( vec3 p ) {
			vec3 lp = p - uCenter;
			float ang = atan( lp.z, lp.x );
			float rr = length( lp.xz );
			// swirl while stirring: rotate the lookup around the axis
			float a2 = ang + uSwirl * ( 1.5 - rr * 0.25 );
			vec3 q = vec3( cos( a2 ) * rr, lp.y, sin( a2 ) * rr );
			float n1 = noise3( vec3( q.x * 0.55, q.y * 0.32 - uTime * 0.12, q.z * 0.55 ) ) * 0.65 + noise3( vec3( q.x * 1.2, q.y * 0.7, q.z * 1.2 ) ) * 0.35;
			float n2 = noise3( q * 1.6 + vec3( 0.0, uTime * 0.35, 0.0 ) );
			float iface = uLayer + ( n1 - 0.5 ) * 2.6 + ( n2 - 0.5 ) * 0.8;
			float layered = uConcTop * smoothstep( iface - uLayerSoft, iface + uLayerSoft, lp.y );
			// sinking plume below the pour point
			vec2 dp = lp.xz - uPlume.xy;
			float plume = uPlume.z * exp( -dot( dp, dp ) * 0.35 ) * smoothstep( 0.35, 0.7, n1 + n2 * 0.4 );
			float layeredTotal = layered + plume;
			float mixed = uConcBase * ( 0.85 + 0.3 * n2 );
			return max( mix( layeredTotal, mixed, uMix ), 0.0 );
		}

		float radiusAt( float ly ) { return uRadius.x + uRadius.y * clamp( ly, 0.0, 30.0 ); }

		float pathLength( vec3 o, vec3 d ) {
			vec3 lo = o - uCenter;
			float r = radiusAt( lo.y );
			float a = dot( d.xz, d.xz );
			float b = dot( lo.xz, d.xz );
			float c = dot( lo.xz, lo.xz ) - r * r;
			float disc = b * b - a * c;
			float tSide = 1e4;
			if ( a > 1e-6 && disc > 0.0 ) tSide = ( - b + sqrt( disc ) ) / a;
			float tY = 1e4;
			if ( d.y < -1e-4 ) tY = ( uBottom - lo.y ) / d.y;
			else if ( d.y > 1e-4 ) tY = ( uLevel - o.y ) / d.y;
			return clamp( min( tSide, tY ), 0.0, 12.0 );
		}

		vec3 surfacePoint( vec3 wp ) {
			vec3 rd = normalize( wp - cameraPosition );
			float tt = ( uLevel - cameraPosition.y ) / min( rd.y, -1e-4 );
			return cameraPosition + rd * tt;
		}
	`;

	const discard = `
		#include <clipping_planes_fragment>
		if ( vWorldPosition.y > uLevel ) discard;
	`;

	const surfaceNormal = `
		#include <normal_fragment_maps>
		if ( ! gl_FrontFacing ) {
			vec3 sp = surfacePoint( vWorldPosition );
			vec3 wn = vec3( 0.0, 1.0, 0.0 );
			vec2 q = sp.xz - uRipple.xy;
			float r = length( q ) + 1e-3;
			wn.xz += ( q / r ) * cos( r * 4.5 - uTime * 16.0 ) * uRipple.z * exp( - r * 0.45 );
			wn.xz += ( vec2( noise3( vec3( sp.xz * 1.4, uTime * 1.3 ) ), noise3( vec3( sp.zx * 1.4 + 7.0, uTime * 1.3 ) ) ) - 0.5 ) * uTurb;
			normal = normalize( ( viewMatrix * vec4( normalize( wn ), 0.0 ) ).xyz );
		}
	`;

	const transmission = `
		material.transmission = transmission;
		material.transmissionAlpha = 1.0;
		vec3 pos = vWorldPosition;
		bool surf = ! gl_FrontFacing;
		if ( surf ) pos = surfacePoint( vWorldPosition );
		vec3 v = normalize( cameraPosition - pos );
		vec3 n = inverseTransformDirection( normal, viewMatrix );
		vec3 rdir = normalize( refract( - v, n, 1.0 / material.ior ) );
		float L = uSimple > 0.5 ? uThick : pathLength( pos, rdir );
		float od = 0.0;
		if ( uConcTop + uConcBase > 0.001 ) {
			for ( int i = 0; i < 5; i ++ ) od += coffeeConc( pos + rdir * ( L * ( float( i ) + 0.5 ) / 5.0 ) );
			od *= L / 5.0;
		}
		vec3 trans = exp( - uSigma * od - uWaterSigma * L );
		vec4 transmitted = getIBLVolumeRefraction(
			n, v, material.roughness, trans, material.specularColor, material.specularF90,
			pos, modelMatrix, viewMatrix, projectionMatrix, material.dispersion, material.ior, max( L, 0.2 ),
			vec3( 1.0 ), 1e20 );
		material.transmissionAlpha = mix( material.transmissionAlpha, transmitted.a, material.transmission );
		totalDiffuse = mix( totalDiffuse, transmitted.rgb, material.transmission );

		// foam on the surface, crema band for the espresso shot
		float rr = length( pos.xz - uCenter.xz );
		float foam = 0.0;
		float fn = noise3( vec3( pos.xz * 2.4, 3.0 ) ) * 0.6 + noise3( vec3( pos.xz * 9.0, 5.0 ) ) * 0.4;
		if ( surf ) {
			float edge = smoothstep( radiusAt( uLevel - uCenter.y ) - 0.9, radiusAt( uLevel - uCenter.y ), rr );
			foam = uFoam * clamp( smoothstep( 0.42, 0.62, fn ) + edge, 0.0, 1.0 );
			foam = max( foam, step( 0.001, uCrema ) );
		} else if ( uCrema > 0.0 ) {
			foam = smoothstep( uLevel - uCrema - 0.1, uLevel - uCrema + 0.1, pos.y );
		}
		vec3 foamCol = uFoamColor * ( 0.75 + 0.5 * fn ) * ( 0.6 + 0.4 * max( dot( n, vec3( 0.0, 1.0, 0.0 ) ), 0.0 ) );
		totalDiffuse = mix( totalDiffuse, foamCol, foam );
		totalSpecular *= 1.0 - 0.75 * foam;
	`;

	const make = (side) => {
		const m = new THREE.MeshPhysicalMaterial({
			color: 0xffffff, metalness: 0, roughness: 0.02, transmission: 1, ior: 1.333, thickness: 1,
			specularIntensity: 1, envMapIntensity: 1.0, side,
		});
		m.onBeforeCompile = (s) => {
			Object.assign(s.uniforms, u);
			s.fragmentShader = s.fragmentShader
				.replace('#include <transmission_pars_fragment>', `${sceneTexPars}\n${decl}`)
				.replace('#include <clipping_planes_fragment>', discard)
				.replace('#include <normal_fragment_maps>', surfaceNormal)
				.replace('#include <transmission_fragment>', transmission);
		};
		return m;
	};

	return { u, front: make(THREE.FrontSide), back: make(THREE.BackSide) };
}

// ---------------------------------------------------------------- ice

export function iceMaterial(normalMap, seed) {
	const m = new THREE.MeshPhysicalMaterial({
		color: 0xffffff, metalness: 0, roughness: 0.05, transmission: 1, ior: 1.31, thickness: 2.2,
		attenuationColor: new THREE.Color(0.86, 0.94, 1.0), attenuationDistance: 7,
		specularIntensity: 1, envMapIntensity: 1.35,
		normalMap, normalScale: new THREE.Vector2(0.45, 0.45),
	});
	m.userData.u = {
		uSeed: { value: seed },
		uCoreR: { value: new THREE.Vector3(0.75, 0.85, 0.75) },
		uCoreDensity: { value: 0.45 },
		uCoreColor: { value: new THREE.Color(1.1, 1.15, 1.2) },
		uWet: { value: 0 },
	};
	m.onBeforeCompile = (s) => {
		Object.assign(s.uniforms, m.userData.u);
		s.vertexShader = header(s.vertexShader, 'varying vec3 vObjPos;')
			.replace('#include <begin_vertex>', '#include <begin_vertex>\nvObjPos = position;');
		s.fragmentShader = header(s.fragmentShader, `varying vec3 vObjPos;\nuniform float uSeed, uCoreDensity, uWet;\nuniform vec3 uCoreR, uCoreColor;\n${GLSL_NOISE}`)
			.replace('#include <transmission_fragment>', `#include <transmission_fragment>
				// cloudy frozen core: chord length through an inner ellipsoid
				vec3 oc = ( inverse( modelMatrix ) * vec4( cameraPosition, 1.0 ) ).xyz;
				vec3 rd = normalize( vObjPos - oc );
				vec3 o2 = vObjPos / uCoreR, d2 = rd / uCoreR;
				float qa = dot( d2, d2 ), qb = dot( o2, d2 ), qc = dot( o2, o2 ) - 1.0;
				float disc = qb * qb - qa * qc;
				float chord = 0.0;
				if ( disc > 0.0 ) { float sq = sqrt( disc ); chord = max( ( - qb + sq ) / qa - max( ( - qb - sq ) / qa, 0.0 ), 0.0 ); }
				vec3 mid = vObjPos + rd * 0.9;
				float veil = 0.55 + 0.9 * fbm3( mid * 1.7 + uSeed * 3.1 );
				float streak = smoothstep( 0.82, 0.95, noise3( vec3( mid.x * 5.0, mid.y * 0.6, mid.z * 5.0 ) + uSeed ) );
				float cloud = 1.0 - exp( - chord * uCoreDensity * veil - streak * 0.5 );
				vec3 coreLit = uCoreColor * ( 0.55 + 0.45 * max( dot( inverseTransformDirection( normal, viewMatrix ), vec3( 0.0, 1.0, 0.0 ) ), 0.0 ) );
				totalDiffuse = mix( totalDiffuse, coreLit, cloud * 0.55 );
			`);
	};
	return m;
}
