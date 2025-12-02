// ============================================================================
// ENHANCED TOON-SHADER OCEAN WITH GERSTNER WAVES
// Features: Realistic wave simulation, toon shading, boat with wind-blown flag,
//           Enscape volumetric clouds, improved underwater with Worley noise
// ============================================================================

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

// Worley (cellular) noise for underwater caustics/patterns.
// We find the closest and second closest feature points in a 3x3 neighborhood.
// The difference between those distances gives a nice cellular pattern.
vec2 worley2D(vec2 p) {
    vec2 id = floor(p); // Integer grid cell 
    vec2 fd = fract(p); // Position within cell
    
    float minDist = 1.0; // Closest feature point distance 
    float secondMinDist = 1.0; // Second closest distance 
    
    // Check neighboring cells to find nearest "feature points"
    for(int y = -1; y <= 1; y++) {
        for(int x = -1; x <= 1; x++) {
            vec2 neighbor = vec2(float(x), float(y));
            
            // Random point in this cell
            vec2 point = hash2D(id + neighbor) * vec2(1.0);
            
            // Distance to feature point
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
    
    // Return both distances for caustic calculation
    return vec2(minDist, secondMinDist); 
}

// ============================================================================
// GERSTNER WAVE SYSTEM
// ============================================================================

// Single directional Gerstner wave calculation
// We use the derivative to "drag" the sample position and get more choppy,
// Gerstner-like waves instead of just flat sine waves.
vec2 wavedx(vec2 position, vec2 direction, float frequency, float timeshift) {
    float x = dot(direction, position) * frequency + timeshift; // Compute wave phase: direction·position scaled by frequency plus time offset
    float wave = exp(sin(x) - 1.0); // Exponential sine creates sharp peaks (Gerstner characteristic)
    float dx = wave * cos(x); // Derivative for horizontal displacement (key Gerstner innovation)
    return vec2(wave, -dx); // Return both height and horizontal drag
}

// Sum of many directional waves to build our main ocean displacement.
// We progressively change frequency, time multiplier and weight to get a fractal-like wave spectrum.
// Accumulate 8 directional waves into complex ocean surface
float getWaves(vec2 position, int iterations, float time) {
    float wavePhaseShift = length(position) * 0.1;
    float iter = 0.0; // Iteration counter for directional rotation
    float frequency = 1.0; // Starting frequency
    float timeMultiplier = 2.0;  // Time scale 
    float weight = 1.0; // Wave amplitude weight
    float sumOfValues = 0.0; // Accumulated wave heights
    float sumOfWeights = 0.0; // Total weight for normalization
    
    // Iterate 8 times to build wave complexity
    for(int i = 0; i < iterations; i++) {
        // Wave direction rotates in circle: (sin, cos) gives unit vector
        vec2 p = vec2(sin(iter), cos(iter));
        // Calculate single wave at current position and direction
        vec2 res = wavedx(position, p, frequency, time * timeMultiplier + wavePhaseShift);
        
        // GERSTNER KEY: Horizontal drag moves sample position
        // This creates choppy peaks instead of smooth sine waves
        position += p * res.y * weight * DRAG_MULT;
        
        sumOfValues += res.x * weight;
        sumOfWeights += weight;
        
        // Decrease weight as we add higher frequency waves
        // Progressive parameter changes for fractal-like spectrum
        weight = mix(weight, 0.0, 0.2);
        frequency *= 1.18;
        timeMultiplier *= 1.07;
        iter += 1232.399963; // Large phase offset for rotation
    }
    
    return sumOfValues / sumOfWeights; // Return normalized sum
}

// Convert our wave accumulation into a final height value with LOD optimization
// The scaling and offset are tuned artistically so the surface looks nice.
float getWaveHeight(vec2 worldPos, float time) {
    float distFromOrigin = length(worldPos);
    
    // LOD factor: 1.0 at <15m, smoothly to 0.0 at >50m
    float lodFactor = 1.0 - smoothstep(15.0, 50.0, distFromOrigin);  
    
    // Get full 8-iteration wave calculation
    float height = getWaves(worldPos, 8, time);
    
    // Scale amplitude based on distance: far waves 10% amplitude
    float amplitudeScale = 0.25 * (0.1 + lodFactor * 0.8);  
    height = height * amplitudeScale - 0.15;
    
    return height;
}


// Calculate surface normal from height field using finite differences
// This turns the scalar height field into a geometric surface normal for lighting.
vec3 calculateWaveNormal(vec2 pos, float time, float epsilon) {
    // Center height
    float H = getWaves(pos, 12, time);
    vec2 ex = vec2(epsilon, 0);
    
    // Three points on surface for cross product
    vec3 a = vec3(pos.x, H, pos.y); // Center
    vec3 b = vec3(pos.x - epsilon, getWaves(pos - ex.xy, 12, time), pos.y); // Left 
    vec3 c = vec3(pos.x, getWaves(pos + ex.yx, 12, time), pos.y + epsilon); // Forward
    
    // Cross product of two tangent vectors gives normal
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

// Apply cel-shaded toon lighting with discrete bands
vec3 applyToonLighting(vec3 baseColor, vec3 normal, vec3 lightDir, vec3 viewDir) {
    float NdotL = max(0.0, dot(normal, lightDir)); // Lambertian diffuse term
    
    // Step 1: toon diffuse: quantize diffuse into 4 discrete bands (toon style)
    float toonDiffuse = toonShading(NdotL, 4); // Bands: 0, 0.25, 0.5, 0.75, 1.0
    
    // Step 2: toon specular using Blinn-Phong style half vector 
    vec3 halfVec = normalize(lightDir + viewDir); 
    float specular = pow(max(0.0, dot(normal, halfVec)), 32.0); // Shininess = 32
    float toonSpecular = toonShading(specular, 2);  // 2 bands: 0, 0.5, 1.0
    
    // Step 3: rim lighting for toon outline effect
    float rim = 1.0 - max(0.0, dot(normal, viewDir)); // Fresnel-like
    rim = pow(rim, 3.0); // Sharpen falloff 
    rim = step(0.65, rim); // Binary threshold 
    
    // Combine into a final toon-lit color
    vec3 ambient = baseColor * 0.4;   // Ambient
    vec3 diffuse = baseColor * (0.5 + toonDiffuse * 0.35); 
    vec3 spec = vec3(1.0) * toonSpecular * 0.4;
    vec3 rimColor = vec3(0.3, 0.35, 0.4) * rim * 0.15;
    
    return clamp(ambient + diffuse + spec + rimColor, 0.0, 1.0);
}

// ============================================================================
// ENHANCED FOAM AND SPRAY
// ============================================================================

// Foam appears on steep wave peaks where water is most turbulent
// Foam along sharp crests: we detect steep areas via normal.y (low y = steep).
// Then we modulate with high-frequency noise so foam breaks up into patches.
float calculateWaveCrestFoam(vec3 worldPos, vec3 normal, float time) {
    float steepness = 1.0 - normal.y;  // Steepness detection: when normal.y is low, the surface is steep (vertical)
    steepness = smoothstep(0.15, 0.35, steepness); // only crests above threshold produce foam
    
    // Two layers of noise at different scales for organic foam pattern
    float foamNoise1 = noise2D(worldPos.xz * 6.0 + time * vec2(0.3, 0.2));
    float foamNoise2 = noise2D(worldPos.xz * 12.0 - time * vec2(0.2, 0.3));
    
    // Multiply noises: creates scattered foam patches (not solid)
    float foam = foamNoise1 * foamNoise2;
    
    // Threshold: only bright areas of noise become foam
    foam = step(0.45, foam);
    
    // Threshold: only bright areas of noise become foam
    return foam * steepness;
}

// "Sea spray" particles: thresholded noise around steep areas,
// so only the big energetic waves throw spray.
float calculateSeaSpray(vec3 worldPos, vec3 normal, float time) {
    float steepness = 1.0 - normal.y; // Steepness check: spray only on steep waves
    if (steepness < 0.2) return 0.0;  // Early exit for flat areas
    
    // High-frequency animated noise for spray particles
    float spray = noise2D(worldPos.xz * 8.0 + time * vec2(0.8, 0.6));
    // Add second layer at different scale and direction
    spray += noise2D(worldPos.xz * 16.0 - time * vec2(0.5, 0.7)) * 0.5;
    // High threshold: only the brightest noise spots become spray
    spray = step(0.75, spray);
    // Scale spray by steepness: steeper waves = more spray
    spray *= smoothstep(0.25, 0.4, steepness);
    
    return spray;
}

// Shoreline foam: we fake waves breaking near the "beach" using depth,
// some scrolling noise and a sine pattern to make multiple foam lines.
// Calculate foam intensity based on wave steepness
float calculateShoreFoam(float depth, float time, vec2 worldPos) {
    const float foamThreshold = 1.2; // Maximum depth for foam effect 
    float foamDiff = saturate(depth / foamThreshold);
    
    // Two layers of noise to keep shore foam irregular
    float foamNoise = noise2D(worldPos * 5.0 + time * vec2(0.5, 0.4));
    foamNoise += noise2D(worldPos * 10.0 - time * vec2(0.3, 0.5)) * 0.5;
    
    // Animated foam stripes sliding in/out with time
    float foamLines = sin((foamDiff - time * 0.8) * 8.0 * PI);
    // Convert sine to positive range and fade with depth
    foamLines = saturate(foamLines) * (1.0 - foamDiff);
    
    // Threshold noise modulated by foam lines
    float foam = step(0.5 - foamLines * 0.3, foamNoise);
    // Fade out foam as depth increases
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
    // Initialize ray marching
    float t = 0.1;
    const int maxSteps = 180;  
    bool hit = false; // Surface hit flag
    vec3 hitPos;
    
    // Ray march to find water surface
    for (int i = 0; i < maxSteps; i++) {
        hitPos = ro + rd * t; // Current position along ray 
        
        float waveHeight = getWaveHeight(hitPos.xz, time); // Get wave height at this position
        float diff = hitPos.y - waveHeight; // Distance to surface
        
        // Adaptive epsilon: tighter near, looser far
        float adaptiveEpsilon = 0.005 + t * 0.0002; 
        
        // Check if we hit the surface
        if (abs(diff) < adaptiveEpsilon) {  
            hit = true;
            break;
        }
        
        // Step size based on distance to surface
        float stepSize = max(0.008, abs(diff) * 0.25); 
        t += stepSize;
       
        if (t > 120.0) break; // Max distance cutoff 
    }
    
    // If no hit, return background
    // Also the key to solve the horizon issue
    if (!hit) {
        return skyColor;
    }
    
    float distance = t;
    
    // Calculate surface normal with distance-adaptive detail
    float normalEps = 0.015 + distance * 0.0001;
    vec3 normal = calculateWaveNormal(hitPos.xz, time, normalEps);
    
    float depth = min(t * 0.2, 8.0);
    
    // ================== STRONGER MICRO PERTURBATION ==================
    // Add micro-scale surface detail using 3-layer Perlin noise
    // It makes the water surface look more detailed without changing its height.
    vec2 microUV = hitPos.xz * 8.0 + iTime * 0.6;   // // Base frequency 8.0, animated

    // Three octaves of noise at different frequencies
    float n1 = noise2D(microUV);   // Base layer
    float n2 = noise2D(microUV * 2.7 + 13.1); // Mid detail  
    float n3 = noise2D(microUV * 5.1 - 7.3); // Fine detail 

    // Combine noise layers: primary + secondary + tertiary
    vec2 m = vec2(n1, n2) * 2.0 - 1.0; // Remap to [-1, 1] 
    m += (n3 * 2.0 - 1.0) * 0.5; // Add fine detail at half strength

    // Perturbation strength: how much normals are affected
    float microStrength = 0.1; // Eg: 0.2 = subtle, 0.5 = very wobbly

    vec3 microNormal = normalize(vec3(
        m.x * microStrength,  // X tilt 
        1.0,                  // Y stays vertical
        m.y * microStrength   // Z tilt 
    ));

    // Blend more towards microNormal for stronger perturbation.
    float blend = 0.15; // Eg: 0.3 subtle, 0.8 crazy
    normal = normalize(mix(normal, microNormal, blend));
    // =====================================================
    
    // IMPROVED UNDERWATER COLORS with better depth transition
    vec3 shallowColor = vec3(0.5, 0.75, 0.95);     
    vec3 midColor = vec3(0.3, 0.65, 0.85);          
    vec3 deepColor = vec3(0.15, 0.45, 0.7);         
    vec3 veryDeepColor = vec3(0.08, 0.25, 0.5);     

    // Worley Caustics (Underwater Light Patterns)   
    vec2 worley = worley2D(hitPos.xz * 2.0 + time * 0.15); // Small-scale caustics for surface detail  
    float caustics = worley.y - worley.x; // Caustic intensity from distance difference (creates cellular pattern)
    caustics = smoothstep(0.1, 0.4, caustics) * 0.6; // Remap caustics to visible range and reduce intensity

    // Large-scale pattern for deep water variation
    vec2 worleyLarge = worley2D(hitPos.xz * 0.2 + time * 0.03);
    float deepPattern = smoothstep(0.3, 0.7, worleyLarge.x);

    // Depth-Based Color Gradient 
    vec3 baseColor;
    float depthT = smoothstep(0.0, 6.0, depth); // Depth factor: 0 at surface, 1 at 6 meters deep

    // Three-tier color transition based on depth
    baseColor = mix(shallowColor, midColor, smoothstep(0.0, 0.3, depthT)); // Shallow (0-30% depth): shallowColor -> midColor
    baseColor = mix(baseColor, deepColor, smoothstep(0.3, 0.7, depthT)); // Mid-depth (30-70% depth): midColor -> deepColor
    baseColor = mix(baseColor, veryDeepColor, smoothstep(0.7, 1.0, depthT)); // Deep water (70-100% depth): deepColor -> veryDeepColor

    // Caustics fade out with depth (strongest at surface)
    float causticsStrength = 1.0 - smoothstep(0.0, 1.5, depth);

    // Adjust caustics contrast
    caustics = pow(caustics, 0.8);
    // Color (0.25, 0.3, 0.35) gives underwater light feel
    // Add caustics as bright blue-cyan additive light
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
    
    // Fresnel: view-angle dependent reflectivity
    // We use Fresnel effect to control water reflectivity based on view angle. The formula is one minus dot product of normal and view direction, raised to power three. At grazing angles - looking across the water - Fresnel approaches one, giving strong reflections. Looking straight down, Fresnel approaches zero, showing more of the underwater color. 
    // This creates realistic water appearance where you see more reflections at the horizon and more transparency when looking down.
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
    
    // Distance-based fade to horizon
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