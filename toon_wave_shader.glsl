#define PI 3.14159265359
#define DRAG_MULT 0.28

float saturate(float x) {
    return clamp(x, 0.0, 1.0);
}

// ============================================================================
// NOISE FUNCTIONS
// ============================================================================

// Simple 2D hash: turns a 2D point into a pseudo-random value in [0,1]
// We use this as our "random number generator" for Perlin and Worley noise.
float hash2D(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
}

// 1D hash version – same idea as hash2D but with a float input.
// Not heavily used here, but handy for any scalar randomization.
float hash(float n) {
    return fract(sin(n) * 43758.5453);
}

// 2D Perlin(-ish) value noise: smooth value noise with quintic interpolation.
// We compute noise on the corners of the cell and then smoothly blend.
float noise2D(vec2 p) {
    vec2 i = floor(p);     // integer cell coordinates
    vec2 f = fract(p);     // local position inside the cell [0,1)
    
    // Smoothstep-like interpolation curve (6t^5 - 15t^4 + 10t^3) so we
    // don't get visible grid edges.
    vec2 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    
    // Corner random values using hash2D
    float a = hash2D(i);
    float b = hash2D(i + vec2(1.0, 0.0));
    float c = hash2D(i + vec2(0.0, 1.0));
    float d = hash2D(i + vec2(1.0, 1.0));
    
    // Bilinear interpolation with our smooth curve u
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Worley noise (a.k.a. cellular noise) for underwater caustics/patterns.
// We find the closest and second closest feature points in a 3x3 neighborhood.
// The difference between those distances gives a nice cellular pattern.
vec2 worley2D(vec2 p) {
    vec2 id = floor(p);
    vec2 fd = fract(p);
    
    float minDist = 1.0;
    float secondMinDist = 1.0;
    
    // Check neighboring cells to find nearest "feature points"
    for(int y = -1; y <= 1; y++) {
        for(int x = -1; x <= 1; x++) {
            vec2 neighbor = vec2(float(x), float(y));
            // Each cell has one pseudo-random point in it
            vec2 point = hash2D(id + neighbor) * vec2(1.0);
            
            float dist = length(neighbor + point - fd);
            
            // Track closest and second-closest distances
            if(dist < minDist) {
                secondMinDist = minDist;
                minDist = dist;
            } else if(dist < secondMinDist) {
                secondMinDist = dist;
            }
        }
    }
    
    return vec2(minDist, secondMinDist);
}

// ============================================================================
// GERSTNER WAVE SYSTEM
// ============================================================================

// Single directional wave: returns (height, derivative along direction).
// We use the derivative to "drag" the sample position and get more choppy,
// Gerstner-like waves instead of just flat sine waves.
vec2 wavedx(vec2 position, vec2 direction, float frequency, float timeshift) {
    float x = dot(direction, position) * frequency + timeshift;
    float wave = exp(sin(x) - 1.0);
    float dx = wave * cos(x);
    return vec2(wave, -dx);
}

// Sum of many directional waves to build our main ocean displacement.
// We progressively change frequency, time multiplier and weight to get
// a fractal-like wave spectrum.
float getWaves(vec2 position, int iterations, float time) {
    float wavePhaseShift = length(position) * 0.1;
    float iter = 0.0;
    float frequency = 1.0;
    float timeMultiplier = 2.0;
    float weight = 1.0;
    float sumOfValues = 0.0;
    float sumOfWeights = 0.0;
    
    for(int i = 0; i < iterations; i++) {
        // Wave direction on a circle
        vec2 p = vec2(sin(iter), cos(iter));
        vec2 res = wavedx(position, p, frequency, time * timeMultiplier + wavePhaseShift);
        
        // Move sampling position along derivative to simulate choppy/Gerstner waves
        position += p * res.y * weight * DRAG_MULT;
        
        sumOfValues += res.x * weight;
        sumOfWeights += weight;
        
        // Decrease weight as we add higher frequency waves
        weight = mix(weight, 0.0, 0.2);
        frequency *= 1.18;
        timeMultiplier *= 1.07;
        iter += 1232.399963; // Just a big random phase offset
    }
    
    return sumOfValues / sumOfWeights;
}

// Convert our wave accumulation into a final height value.
// The scaling and offset are tuned artistically so the surface looks nice.
float getWaveHeight(vec2 worldPos, float time) {
    float distFromOrigin = length(worldPos);
    
    float lodFactor = 1.0 - smoothstep(15.0, 50.0, distFromOrigin);  
    
    float height = getWaves(worldPos, 8, time);
   
    float amplitudeScale = 0.25 * (0.1 + lodFactor * 0.8);  
    height = height * amplitudeScale - 0.15;
    
    return height;
}

// Approximate normal by sampling nearby heights and taking a cross product.
// This turns the scalar height field into a geometric surface normal for lighting.
vec3 calculateWaveNormal(vec2 pos, float time, float epsilon) {
    float H = getWaves(pos, 12, time);
    vec2 ex = vec2(epsilon, 0);
    
    vec3 a = vec3(pos.x, H, pos.y);
    vec3 b = vec3(pos.x - epsilon, getWaves(pos - ex.xy, 12, time), pos.y);
    vec3 c = vec3(pos.x, getWaves(pos + ex.yx, 12, time), pos.y + epsilon);
    
    return normalize(cross(a - b, a - c));
}

// ============================================================================
// TOON SHADING UTILITIES
// ============================================================================

// Core toon quantization: snap intensity into N discrete bands instead of smooth shading.
float toonShading(float intensity, int bands) {
    float bandSize = 1.0 / float(bands);
    float band = floor(intensity / bandSize);
    return band * bandSize;
}

// Toon lighting: diffuse + specular + rim, all quantized to create a stylized look.
vec3 applyToonLighting(vec3 baseColor, vec3 normal, vec3 lightDir, vec3 viewDir) {
    float NdotL = max(0.0, dot(normal, lightDir));
    
    // Step 1: toon diffuse
    float toonDiffuse = toonShading(NdotL, 4);
    
    // Step 2: toon specular using Blinn-Phong style half vector
    vec3 halfVec = normalize(lightDir + viewDir);
    float specular = pow(max(0.0, dot(normal, halfVec)), 32.0);
    float toonSpecular = toonShading(specular, 2);
    
    // Step 3: rim light (used as a "toon outline" on the lit side)
    float rim = 1.0 - max(0.0, dot(normal, viewDir));
    rim = pow(rim, 3.0);
    rim = step(0.65, rim);
    
    // Combine into a final toon-lit color
    vec3 ambient = baseColor * 0.4;
    vec3 diffuse = baseColor * (0.5 + toonDiffuse * 0.35);
    vec3 spec = vec3(1.0) * toonSpecular * 0.4;
    vec3 rimColor = vec3(0.3, 0.35, 0.4) * rim * 0.15;
    
    return clamp(ambient + diffuse + spec + rimColor, 0.0, 1.0);
}

// ============================================================================
// ENHANCED FOAM AND SPRAY
// ============================================================================

// Foam along sharp crests: we detect steep areas via normal.y (low y = steep).
// Then we modulate with high-frequency noise so foam breaks up into patches.
float calculateWaveCrestFoam(vec3 worldPos, vec3 normal, float time) {
    float steepness = 1.0 - normal.y;
    steepness = smoothstep(0.15, 0.35, steepness);
    
    float foamNoise1 = noise2D(worldPos.xz * 6.0 + time * vec2(0.3, 0.2));
    float foamNoise2 = noise2D(worldPos.xz * 12.0 - time * vec2(0.2, 0.3));
    
    float foam = foamNoise1 * foamNoise2;
    foam = step(0.45, foam);
    
    return foam * steepness;
}

// "Sea spray" particles: thresholded noise around steep areas,
// so only the big energetic waves throw spray.
float calculateSeaSpray(vec3 worldPos, vec3 normal, float time) {
    float steepness = 1.0 - normal.y;
    if (steepness < 0.2) return 0.0;
    
    float spray = noise2D(worldPos.xz * 8.0 + time * vec2(0.8, 0.6));
    spray += noise2D(worldPos.xz * 16.0 - time * vec2(0.5, 0.7)) * 0.5;
    spray = step(0.75, spray);
    spray *= smoothstep(0.25, 0.4, steepness);
    
    return spray;
}

// Shoreline foam: we fake waves breaking near the "beach" using depth,
// some scrolling noise and a sine pattern to make multiple foam lines.
float calculateShoreFoam(float depth, float time, vec2 worldPos) {
    const float foamThreshold = 1.2;
    float foamDiff = saturate(depth / foamThreshold);
    
    // Two layers of noise to keep shore foam irregular
    float foamNoise = noise2D(worldPos * 5.0 + time * vec2(0.5, 0.4));
    foamNoise += noise2D(worldPos * 10.0 - time * vec2(0.3, 0.5)) * 0.5;
    
    // Animated foam stripes sliding in/out with time
    float foamLines = sin((foamDiff - time * 0.8) * 8.0 * PI);
    foamLines = saturate(foamLines) * (1.0 - foamDiff);
    
    float foam = step(0.5 - foamLines * 0.3, foamNoise);
    return foam * (1.0 - saturate(depth / foamThreshold));
}

// ============================================================================
// 2D CLOUD SHADER (Simple and beautiful)
// ============================================================================

// Cloud parameters, mostly borrowed from classic 2D sky shaders and tweaked
// for a toon vibe later on.
const float cloudscale = 1.1;
const float cloudSpeed = 0.03;
const float clouddark = 0.5;
const float cloudlight = 0.3;
const float cloudcover = 0.2;
const float cloudalpha = 8.0;
const float skytint = 0.5;
const vec3 skycolour1 = vec3(0.2, 0.4, 0.6);
const vec3 skycolour2 = vec3(0.4, 0.7, 1.0);
const mat2 cloudMatrix = mat2(1.6, 1.2, -1.2, 1.6);

// Hash for cloud noise: similar idea as earlier hash, but produces a vec2.
vec2 hashCloud(vec2 p) {
    p = vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)));
    return -1.0 + 2.0 * fract(sin(p) * 43758.5453123);
}

// 2D simplex-style noise function used as the base building block for clouds.
// This is smoother and cheaper than naive grid noise.
float noiseCloud(in vec2 p) {
    const float K1 = 0.366025404; // (sqrt(3)-1)/2;
    const float K2 = 0.211324865; // (3-sqrt(3))/6;
    vec2 i = floor(p + (p.x + p.y) * K1);
    vec2 a = p - i + (i.x + i.y) * K2;
    vec2 o = (a.x > a.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
    vec2 b = a - o + K2;
    vec2 c = a - 1.0 + 2.0 * K2;
    vec3 h = max(0.5 - vec3(dot(a, a), dot(b, b), dot(c, c)), 0.0);
    vec3 n = h * h * h * h * vec3(dot(a, hashCloud(i + 0.0)), dot(b, hashCloud(i + o)), dot(c, hashCloud(i + 1.0)));
    return dot(n, vec3(70.0));
}

// fBm (fractal Brownian motion) for clouds: sum several octaves of noiseCloud.
// Higher octaves get smaller and weaker to create soft, billowy shapes.
float fbmCloud(vec2 n) {
    float total = 0.0, amplitude = 0.1;
    for (int i = 0; i < 7; i++) {
        total += noiseCloud(n) * amplitude;
        n = cloudMatrix * n;
        amplitude *= 0.4;
    }
    return total;
}

// Main 2D cloud rendering, heavily inspired by Enscape-style clouds.
// We build several layers: shape, detail, and coloring, then quantize for toon.
vec3 renderClouds(vec2 fragCoord) {
    vec2 p = fragCoord.xy / iResolution.xy;
    vec2 uv = p * vec2(iResolution.x / iResolution.y, 1.0);
    float time = iTime * cloudSpeed;
    float q = fbmCloud(uv * cloudscale * 0.5);
    
    // Ridged noise shape (absolute value) gives sharper edges to cloud forms.
    float r = 0.0;
    uv *= cloudscale;
    uv -= q - time;
    float weight = 0.8;
    for (int i = 0; i < 8; i++) {
        r += abs(weight * noiseCloud(uv));
        uv = cloudMatrix * uv + time;
        weight *= 0.7;
    }
    
    // Second layer of noise for softer internal variation.
    float f = 0.0;
    uv = p * vec2(iResolution.x / iResolution.y, 1.0);
    uv *= cloudscale;
    uv -= q - time;
    weight = 0.7;
    for (int i = 0; i < 8; i++) {
        f += weight * noiseCloud(uv);
        uv = cloudMatrix * uv + time;
        weight *= 0.6;
    }
    
    f *= r + f;
    
    // Extra color modulation noise
    float c = 0.0;
    time = iTime * cloudSpeed * 2.0;
    uv = p * vec2(iResolution.x / iResolution.y, 1.0);
    uv *= cloudscale * 2.0;
    uv -= q - time;
    weight = 0.4;
    for (int i = 0; i < 7; i++) {
        c += weight * noiseCloud(uv);
        uv = cloudMatrix * uv + time;
        weight *= 0.6;
    }
    
    // Ridged color detail
    float c1 = 0.0;
    time = iTime * cloudSpeed * 3.0;
    uv = p * vec2(iResolution.x / iResolution.y, 1.0);
    uv *= cloudscale * 3.0;
    uv -= q - time;
    weight = 0.4;
    for (int i = 0; i < 7; i++) {
        c1 += abs(weight * noiseCloud(uv));
        uv = cloudMatrix * uv + time;
        weight *= 0.6;
    }
    
    c += c1;
    
    // Vertical sky color gradient (horizon to zenith)
    vec3 skycolour = mix(skycolour2, skycolour1, p.y);
    
    // TOON SHADER THRESHOLD for clouds - quantize into discrete bands
    float cloudLuminance = clamp((clouddark + cloudlight * c), 0.0, 1.0);
    
    // Manually map luminance to a few fixed levels so clouds look cartoony.
    if (cloudLuminance > 0.65) {
        cloudLuminance = 0.8;  // Bright
    } else if (cloudLuminance > 0.35) {
        cloudLuminance = 0.65;  // Mid
    } else if (cloudLuminance > 0.25) {
        cloudLuminance = 0.45;  // Mid
    } else {
        cloudLuminance = 0.25; // Shadow
    }
    
    vec3 cloudcolour = vec3(1.1, 1.1, 0.9) * cloudLuminance;
    
    f = cloudcover + cloudalpha * f * r;
    
    // Also quantize cloud alpha for sharper edges (less soft, more graphic).
    float cloudAlpha = clamp(f + c, 0.0, 1.0);
    cloudAlpha = smoothstep(0.3, 0.7, cloudAlpha);
    
    // Blend clouds with sky
    vec3 result = mix(skycolour, clamp(skytint * skycolour + cloudcolour, 0.0, 1.0), cloudAlpha);
    
    return result;
}

// ============================================================================
// BOAT WITH WIND-BLOWN FLAG
// ============================================================================

// SDF primitives for modeling the boat. We use signed distance functions so
// we can raymarch a complex shape made from simple building blocks.

// Box SDF
float sdBox(vec3 p, vec3 b) {
    vec3 q = abs(p) - b;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}

// Cylinder SDF (infinite in XZ, limited in height via h)
float sdCylinder(vec3 p, float r, float h) {
    vec2 d = abs(vec2(length(p.xz), p.y)) - vec2(r, h);
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}

// Rounded box SDF for softer edges on the hull and cabin.
float sdRoundBox(vec3 p, vec3 b, float r) {
    vec3 q = abs(p) - b;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0) - r;
}

// Smooth min: blends between two SDFs so we don't get hard seams where shapes meet.
float smin(float a, float b, float k) {
    float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}

// Flag SDF with procedural flutter: we offset the flag geometry by a few sine
// waves that get stronger towards the free end of the flag.
float sdFlag(vec3 p, float time) {
    vec3 flagP = p;
    flagP.y -= 0.75;
    flagP.z -= 0.15;
    flagP.x += 0.08;
    
    // Distance from pole - flag waves more at the free end
    float distFromPole = max(0.0, flagP.x - 0.02);
    float distFactor = (flagP.x - 0.02) * 10.0; // Exponential increase
    
    // Multiple wave frequencies for realistic flutter
    float wave1 = sin(flagP.x * 12.0 + time * 4.5) * 0.025 * distFactor;
    float wave2 = sin(flagP.x * 20.0 - time * 6.0) * 0.015 * distFactor;
    float wave3 = sin(flagP.x * 35.0 + time * 8.0) * 0.008 * distFactor;
    
    // Add some vertical flutter
    float verticalWave = sin(flagP.x * 10.0 + time * 5.0) * 0.02 * distFactor;
    
    // Apply waves
    flagP.y += wave1 + wave2 + verticalWave;
    flagP.z += (wave1 + wave3) * 0.6;
    
    float flag = sdBox(flagP, vec3(0.12, 0.08, 0.002));
    return flag;
}

// Boat SDF composed from hull, cabin, mast, flag, etc.
// Each part is a primitive SDF, combined using smin or min.
float sdBoat(vec3 p, float time) {
    vec3 hullP = p;
    hullP.y += 0.08;
    float hull = sdRoundBox(hullP, vec3(0.9, 0.1, 0.3), 0.05);
    
    vec3 bowP = p;
    bowP.z += 0.35;
    bowP.y += 0.08;
    float bow = sdBox(bowP, vec3(0.5, 0.08, 0.05));
    hull = smin(hull, bow, 0.1);
    
    vec3 sternP = p;
    sternP.z -= 0.35;
    sternP.y += 0.08;
    float stern = sdRoundBox(sternP, vec3(0.7, 0.08, 0.05), 0.03);
    hull = smin(hull, stern, 0.08);
    
    vec3 deckP = p;
    deckP.y -= 0.02;
    float deck = sdBox(deckP, vec3(0.85, 0.02, 0.28));
    hull = smin(hull, deck, 0.04);
    
    vec3 cabinP = p;
    cabinP.y -= 0.2;
    cabinP.z -= 0.15;
    float cabin = sdRoundBox(cabinP, vec3(0.4, 0.15, 0.3), 0.03);
    
    vec3 roofP = p;
    roofP.y -= 0.38;
    roofP.z -= 0.15;
    float roof = sdRoundBox(roofP, vec3(0.42, 0.03, 0.32), 0.02);
    cabin = smin(cabin, roof, 0.05);
    
    vec3 windowP = p;
    windowP.y -= 0.22;
    windowP.z -= 0.15;
    windowP.x -= 0.25;
    float window1 = sdBox(windowP, vec3(0.08, 0.08, 0.32));
    
    windowP.x += 0.5;
    float window2 = sdBox(windowP, vec3(0.08, 0.08, 0.32));
    
    // Carve out windows by subtracting their SDF from the cabin
    cabin = max(cabin, -window1 * 0.5);
    cabin = max(cabin, -window2 * 0.5);
    
    vec3 mastP = p;
    mastP.y -= 0.5;
    mastP.z -= 0.2;
    float mast = sdCylinder(mastP, 0.03, 0.5);
    
    vec3 nestP = p;
    nestP.y -= 0.85;
    nestP.z -= 0.2;
    float nest = sdCylinder(nestP, 0.08, 0.04);
    mast = smin(mast, nest, 0.03);
    
    float flag = sdFlag(p, time);
    
    vec3 stackP = p;
    stackP.y -= 0.45;
    stackP.z += 0.1;
    stackP.x -= 0.2;
    float stack = sdCylinder(stackP, 0.05, 0.08);
    
    float boat = smin(hull, cabin, 0.06);
    boat = min(boat, mast);
    boat = min(boat, flag);
    boat = min(boat, stack);
    
    return boat;
}

// Material ID lookup for the boat: we re-test SDFs to see which piece is closest,
// so we can assign different colors (hull/cabin/mast/flag).
int getBoatMaterialID(vec3 p, float time) {
    vec3 hullP = p;
    hullP.y += 0.08;
    float hull = sdRoundBox(hullP, vec3(0.9, 0.1, 0.3), 0.05);
    
    vec3 cabinP = p;
    cabinP.y -= 0.2;
    cabinP.z -= 0.15;
    float cabin = sdRoundBox(cabinP, vec3(0.4, 0.15, 0.3), 0.03);
    
    vec3 mastP = p;
    mastP.y -= 0.5;
    mastP.z -= 0.2;
    float mast = sdCylinder(mastP, 0.03, 0.5);
    
    float flag = sdFlag(p, time);
    
    if (flag < hull && flag < cabin && flag < mast) return 3;
    if (mast < hull && mast < cabin) return 2;
    if (cabin < hull) return 1;
    return 0;
}

// Estimate boat normal by sampling the SDF in x,y,z directions.
// Standard trick for SDF-based raymarching.
vec3 estimateNormalBoat(vec3 p, float time) {
    float eps = 0.001;
    float dx = sdBoat(p + vec3(eps, 0, 0), time) - sdBoat(p - vec3(eps, 0, 0), time);
    float dy = sdBoat(p + vec3(0, eps, 0), time) - sdBoat(p - vec3(0, eps, 0), time);
    float dz = sdBoat(p + vec3(0, 0, eps), time) - sdBoat(p - vec3(0, 0, eps), time);
    return normalize(vec3(dx, dy, dz));
}

// Raymarch the boat SDF using sphere tracing (march by the distance itself).
// If we get close enough (d < EPS), we treat that as a hit.
float intersectBoatSDF(vec3 ro, vec3 rd, vec3 center, float time) {
    float t = 0.0;
    const int MAX_STEPS = 100;
    const float EPS = 0.001;
    const float MAX_DIST = 80.0;
    for (int i = 0; i < MAX_STEPS; i++) {
        vec3 p = ro + rd * t;
        float d = sdBoat(p - center, time);
        if (d < EPS) return t;
        t += d;
        if (t > MAX_DIST) break;
    }
    return -1.0;
}

// Boat shading: pick base color from material ID and then apply our toon lighting.
vec3 shadeBoat(vec3 pos, vec3 normal, vec3 viewDir, vec3 boatCenter, float time) {
    vec3 lightDir = normalize(vec3(0.7, 0.6, -0.3));
    
    int matID = getBoatMaterialID(pos - boatCenter, time);
    
    vec3 base;
    if (matID == 0) {
        base = vec3(0.85, 0.25, 0.18);
    } else if (matID == 1) {
        base = vec3(0.98, 0.95, 0.88);
    } else if (matID == 2) {
        base = vec3(0.45, 0.32, 0.22);
    } else {
        base = vec3(0.95, 0.12, 0.12);
    }
    
    return applyToonLighting(base, normal, lightDir, viewDir);
}

// ============================================================================
// IMPROVED WATER RENDERING WITH WORLEY NOISE
// ============================================================================

// Simple shadow check by marching from water surface toward the light and
// testing the boat SDF. If we find the boat, we darken that pixel.
float getBoatShadow(vec3 worldPos, vec3 boatCenter, vec3 lightDir, float time) {
    vec3 shadowRayOrigin = worldPos + vec3(0.0, 0.01, 0.0);
    vec3 shadowRayDir = lightDir;
    
    float t = 0.0;
    const int steps = 20;
    for (int i = 0; i < steps; i++) {
        vec3 p = shadowRayOrigin + shadowRayDir * t;
        float d = sdBoat(p - boatCenter, time);
        if (d < 0.05) return 0.3;
        t += max(0.05, d);
        if (t > 3.0) break;
    }
    return 1.0;
}

// Reflection raymarching: we reflect the view direction on the water normal,
// then march against a mirrored boat to approximate a reflection in the water.
vec3 getBoatReflectionColor(vec3 worldPos, vec3 boatCenter, vec3 viewDir, vec3 normal, float time) {
    vec3 reflectDir = reflect(viewDir, normal);
    
    // Mirror boat across water plane (y=0) for reflection
    vec3 mirroredBoatCenter = boatCenter;
    mirroredBoatCenter.y = -mirroredBoatCenter.y;
    
    float t = 0.0;
    const int steps = 35;
    for (int i = 0; i < steps; i++) {
        vec3 p = worldPos + reflectDir * t;
        
        float d = sdBoat(p - mirroredBoatCenter, time);
        if (d < 0.08) {
            int matID = getBoatMaterialID(p - mirroredBoatCenter, time);
            
            vec3 refNormal = estimateNormalBoat(p - mirroredBoatCenter, time);
            vec3 lightDir = normalize(vec3(0.7, 0.6, -0.3));
            float refLight = max(0.3, dot(refNormal, lightDir));
            
            // Slightly darker than actual boat so reflection feels softer
            if (matID == 0) return vec3(0.75, 0.22, 0.16) * refLight;
            if (matID == 1) return vec3(0.9, 0.88, 0.82) * refLight;
            if (matID == 2) return vec3(0.4, 0.28, 0.2) * refLight;
            return vec3(0.92, 0.15, 0.12) * refLight;
        }
        t += max(0.04, d);
        if (t > 8.0) break;
    }
    return vec3(0.0);
}

// Main water rendering: raymarch to intersect the heightfield, then shade using
// toon lighting, Worley caustics, foam, reflections and Fresnel.
vec3 renderWater(vec3 ro, vec3 rd, vec3 skyColor, vec3 boatCenter, float time) {
    float t = 0.1;
    const int maxSteps = 180;  
    bool hit = false;
    vec3 hitPos;
    
    // March along the ray until we cross the wave surface.
    for (int i = 0; i < maxSteps; i++) {
        hitPos = ro + rd * t;
        float waveHeight = getWaveHeight(hitPos.xz, time);
        float diff = hitPos.y - waveHeight;
        
        float adaptiveEpsilon = 0.005 + t * 0.0002; 
        
        if (abs(diff) < adaptiveEpsilon) {  
            hit = true;
            break;
        }
        
        float stepSize = max(0.008, abs(diff) * 0.25); 
        t += stepSize;
       
        if (t > 120.0) break;  
    }
    
    // key to solve the horizon issue
    if (!hit) {
        return skyColor;
    }
    
    float distance = t;
    
    float normalEps = 0.015 + distance * 0.0001;
    vec3 normal = calculateWaveNormal(hitPos.xz, time, normalEps);
    
    float depth = min(t * 0.2, 8.0);
    
    // ================== STRONGER MICRO PERTURBATION ==================
    // Here we add a "fake normal map" using multiple layers of Perlin noise.
    // It makes the water surface look more detailed without changing its height.
    vec2 microUV = hitPos.xz * 8.0 + iTime * 0.6;   // higher freq, faster move

    float n1 = noise2D(microUV);
    float n2 = noise2D(microUV * 2.7 + 13.1);
    float n3 = noise2D(microUV * 5.1 - 7.3);

    // Combine noises for richer pattern in [-1,1]
    vec2 m = vec2(n1, n2) * 2.0 - 1.0;
    m += (n3 * 2.0 - 1.0) * 0.5;     // extra detail

    // How strong the micro bumps are
    float microStrength = 0.1;      // 0.2 = subtle, 0.5 = very wobbly

    vec3 microNormal = normalize(vec3(
        m.x * microStrength,
        1.0,
        m.y * microStrength
    ));

    // Blend more towards microNormal for stronger perturbation.
    // This basically "tilts" the normal by small noisy amounts.
    float blend = 0.15;               // 0.3 subtle, 0.8 crazy
    normal = normalize(mix(normal, microNormal, blend));
    // =====================================================
    
    // IMPROVED UNDERWATER COLORS with better depth transition
    vec3 shallowColor = vec3(0.5, 0.75, 0.95);     
    vec3 midColor = vec3(0.3, 0.65, 0.85);          
    vec3 deepColor = vec3(0.15, 0.45, 0.7);         
    vec3 veryDeepColor = vec3(0.08, 0.25, 0.5);     

    // Worley caustics 
    vec2 worley = worley2D(hitPos.xz * 2.0 + time * 0.15);  
    float caustics = worley.y - worley.x;
    caustics = smoothstep(0.1, 0.4, caustics) * 0.6;  

    vec2 worleyLarge = worley2D(hitPos.xz * 0.2 + time * 0.03);
    float deepPattern = smoothstep(0.3, 0.7, worleyLarge.x);

    vec3 baseColor;
    float depthT = smoothstep(0.0, 6.0, depth); 

    baseColor = mix(shallowColor, midColor, smoothstep(0.0, 0.3, depthT));
    baseColor = mix(baseColor, deepColor, smoothstep(0.3, 0.7, depthT));
    baseColor = mix(baseColor, veryDeepColor, smoothstep(0.7, 1.0, depthT));

    float causticsStrength = 1.0 - smoothstep(0.0, 1.5, depth);

    caustics = pow(caustics, 0.8);  
    baseColor += vec3(0.25, 0.3, 0.35) * caustics * causticsStrength;  

    
    // Apply toon lighting to the water surface.
    vec3 lightDir = normalize(vec3(0.7, 0.6, -0.3));
    vec3 viewDir = -rd;
    vec3 shadedColor = applyToonLighting(baseColor, normal, lightDir, viewDir);
    
    // Boat shadow
    float shadow = getBoatShadow(hitPos, boatCenter, lightDir, time);
    if (shadow < 1.0) {
        shadedColor *= (0.6 + shadow * 0.4);
    }
    
    // Boat reflection
    vec3 reflectionColor = getBoatReflectionColor(hitPos, boatCenter, rd, normal, time);
    if (length(reflectionColor) > 0.0) {
        float reflectionStrength = 0.65;
        // Slight distortion in reflection using noise so it looks wavy.
        float distortion = noise2D(hitPos.xz * 3.0 + time * 0.5) * 0.1;
        reflectionStrength *= (1.0 - distortion);
        
        shadedColor = mix(shadedColor, reflectionColor, reflectionStrength);
        
        float ripple = smoothstep(0.5, 0.7, noise2D(hitPos.xz * 8.0 + time));
        shadedColor += reflectionColor * ripple * 0.2;
    }
    
    // Fresnel factor: view-angle dependent highlight.
// We quantize it with toonShading so the reflection rim also feels stylized.
    float fresnel = pow(1.0 - max(0.0, dot(normal, viewDir)), 3.0);
    fresnel = toonShading(fresnel, 2);
    shadedColor = mix(shadedColor, skyColor * 0.8, fresnel * 0.15);
    
    // Foam system: combine shoreline, crest foam, and spray to get final foam mask.
    float shoreFoam = calculateShoreFoam(depth, time, hitPos.xz);
    float crestFoam = calculateWaveCrestFoam(hitPos, normal, time);
    float spray = calculateSeaSpray(hitPos, normal, time);
    
    float totalFoam = max(shoreFoam, max(crestFoam, spray));
    
    vec3 foamColor = vec3(0.95, 0.98, 1.0);
    shadedColor = mix(shadedColor, foamColor, totalFoam * 0.85);
    shadedColor += vec3(1.0) * spray * 0.3;
    
    // anti-aliasing
    float distanceFade = smoothstep(70.0, 115.0, distance); 
    shadedColor = mix(shadedColor, skyColor, distanceFade * 0.5);
    
    return shadedColor;
}

// Intersect the ray with the water surface by marching along and checking when
// the y-position matches the wave height within a small epsilon.
float intersectWater(vec3 ro, vec3 rd, out vec3 outHitPos) {
    float t = 0.1;
    const int maxSteps = 100;
    const float epsilon = 0.05;
    for (int i = 0; i < maxSteps; i++) {
        vec3 pos = ro + rd * t;
        float waveHeight = getWaveHeight(pos.xz, iTime);
        float diff = pos.y - waveHeight;
        if (diff < epsilon && diff > -epsilon) {
            outHitPos = pos;
            return t;
        }
        t += max(0.02, abs(diff) * 0.5);
        if (t > 80.0) break;
    }
    return -1.0;
}

// ============================================================================
// MAIN IMAGE
// ============================================================================

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    // Convert pixel into NDC-like coordinates
    vec2 uv = (fragCoord * 2.0 - iResolution.xy) / iResolution.y;
    
    // Camera setup: simple forward-facing camera looking down +Z.
    vec3 ro = vec3(0.0, 1.5, -4.0);
    vec3 rd = normalize(vec3(uv.x, uv.y, 1.0));
    
    // Beautiful 2D cloud rendering with blue sky
    vec3 cloudColor = renderClouds(fragCoord);
    vec3 finalColor = cloudColor;
    
    // Boat setup with wave following:
    // we make the boat drift and bob up and down based on Perlin noise.
    vec2 objXZ = vec2(0.3, -1.5);
    float driftSpeed = 0.15;
    float driftAmt = 0.5;
    
    objXZ += vec2(sin(iTime * driftSpeed) * driftAmt,
                  cos(iTime * driftSpeed * 0.7) * (driftAmt * 0.6));
    
    float objWaveY = getWaveHeight(objXZ, iTime);
    float bob = noise2D(objXZ * 1.5 + iTime * 0.8) * 0.05;
    
    vec3 boatCenter = vec3(objXZ.x, objWaveY + 0.15 + bob, objXZ.y);
    
    // Water intersection
    vec3 waterHitPos;
    float tWater = intersectWater(ro, rd, waterHitPos);
    
    // Boat intersection via SDF raymarching
    float tBoat = intersectBoatSDF(ro, rd, boatCenter, iTime);
    
    // Render boat if visible (and in front of water hit)
    if (tBoat > 0.0 && (tWater < 0.0 || tBoat < tWater)) {
        vec3 surfPos = ro + rd * tBoat;
        vec3 surfNormal = estimateNormalBoat(surfPos - boatCenter, iTime);
        vec3 viewDir = -rd;
        vec3 boatColor = shadeBoat(surfPos, surfNormal, viewDir, boatCenter, iTime);
        
        finalColor = boatColor;
    } else {
        // Render water - only below the horizon (rd.y < 0.0)
        if (rd.y < 0.5) {  
            vec3 waterColor = renderWater(ro, rd, cloudColor, boatCenter, iTime);
            if (dot(waterColor, waterColor) > 0.0) {
                finalColor = waterColor;
            }
        }
    }
    
    fragColor = vec4(finalColor, 1.0);
}