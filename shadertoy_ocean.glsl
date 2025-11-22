// ============================================================================
// STYLIZED OCEAN SHADER FOR SHADERTOY
// Based on: https://halisavakis.com/my-take-on-shaders-stylized-water-shader/
// Features: Toon shading, Fresnel effects, animated normal maps, foam
// ============================================================================

#define PI 3.14159265359

// Helper function - clamp to 0-1 range (like HLSL saturate)
float saturate(float x) {
    return clamp(x, 0.0, 1.0);
}

// ============================================================================
// NOISE FUNCTIONS
// ============================================================================

// 2D Hash function
float hash2D(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
}

// 2D Perlin-like noise with anti-aliasing
float noise2D(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    
    // Quintic interpolation for smoother results (reduces aliasing)
    vec2 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    
    float a = hash2D(i);
    float b = hash2D(i + vec2(1.0, 0.0));
    float c = hash2D(i + vec2(0.0, 1.0));
    float d = hash2D(i + vec2(1.0, 1.0));
    
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// 3D Hash function for clouds
vec3 hash3(vec3 p) {
    float x = sin(p.x * 443.897 + p.y * 441.423 + p.z * 437.195) * 43758.5453;
    float y = sin(p.x * 419.123 + p.y * 431.654 + p.z * 421.321) * 43758.5453;
    float z = sin(p.x * 433.789 + p.y * 427.456 + p.z * 439.654) * 43758.5453;
    return fract(vec3(x, y, z));
}

// 3D Perlin noise for clouds
float noise3D(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    vec3 u = f * f * (3.0 - 2.0 * f);
    
    float result = 0.0;
    for (int z = 0; z <= 1; z++) {
        for (int y = 0; y <= 1; y++) {
            for (int x = 0; x <= 1; x++) {
                vec3 corner = i + vec3(x, y, z);
                vec3 h = hash3(corner);
                float wx = (x == 0) ? (1.0 - u.x) : u.x;
                float wy = (y == 0) ? (1.0 - u.y) : u.y;
                float wz = (z == 0) ? (1.0 - u.z) : u.z;
                result += (h.x - 0.5) * wx * wy * wz;
            }
        }
    }
    return result * 0.5 + 0.5;
}

// Fractal Brownian Motion
float fbm(vec3 p, int octaves) {
    float total = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;
    
    for (int i = 0; i < octaves; i++) {
        total += noise3D(p * frequency) * amplitude;
        frequency *= 2.0;
        amplitude *= 0.5;
    }
    return total;
}

// ============================================================================
// WATER HEIGHT DISPLACEMENT - Creates actual up/down waves
// ============================================================================

float getWaveHeight(vec2 worldPos, float time) {
    // Multiple layers of noise for wave height
    float height = 0.0;
    
    // Large waves (primary swell)
    height += noise2D(worldPos * 0.8 + time * vec2(0.15, 0.10)) * 0.15;
    
    // Medium waves
    height += noise2D(worldPos * 1.2 + time * vec2(-0.12, 0.15)) * 0.10;
    
    // Small ripples
    height += noise2D(worldPos * 2.0 + time * vec2(0.08, -0.08)) * 0.05;
    
    // Bias to center around 0
    height = (height - 0.15) * 2.0;
    
    return height;
}

// ============================================================================
// WATER NORMAL MAPS (Simulated)
// ============================================================================

vec3 sampleWaterNormal(vec2 uv, float time, float scale, vec2 panSpeed, float strength, float distance) {
    vec2 offset = time * panSpeed;
    vec2 p = uv * scale + offset;
    
    // Distance-based LOD to reduce aliasing
    float lod = log2(max(1.0, distance * 0.5));
    float h = 0.02 * (1.0 + lod * 0.5);  // Increase sample spacing with distance
    
    float heightCenter = noise2D(p);
    float heightRight = noise2D(p + vec2(h, 0.0));
    float heightUp = noise2D(p + vec2(0.0, h));
    
    // Calculate normal using finite differences
    float dx = (heightRight - heightCenter) / h;
    float dy = (heightUp - heightCenter) / h;
    
    // Reduce strength with distance to prevent aliasing
    float distanceFactor = 1.0 / (1.0 + distance * 0.1);
    float adjustedStrength = strength * distanceFactor;
    
    vec3 normal = normalize(vec3(-dx * adjustedStrength, 1.0, -dy * adjustedStrength));
    return normal;
}

// Blend two normals (additive like in tutorial)
vec3 blendNormals(vec3 n1, vec3 n2) {
    return normalize(vec3(n1.xy + n2.xy, n1.z * n2.z));
}

// Get combined water normal - MUCH MORE DYNAMIC with anti-aliasing
vec3 getWaterNormal(vec2 worldPos, float time, float distance) {
    // Two normal maps with FASTER speeds and MORE strength for visible waves
    // Distance parameter helps reduce aliasing
    vec3 normalA = sampleWaterNormal(worldPos, time, 0.8, vec2(0.15, 0.10), 0.8, distance);
    vec3 normalB = sampleWaterNormal(worldPos, time, 1.2, vec2(-0.12, 0.15), 0.8, distance);
    
    // Add third layer for more wave variety
    vec3 normalC = sampleWaterNormal(worldPos, time, 2.0, vec2(0.08, -0.08), 0.6, distance);
    
    vec3 blended = blendNormals(normalA, normalB);
    blended = blendNormals(blended, normalC);
    
    return blended;
}

// ============================================================================
// TOON SHADING
// ============================================================================

float toonShading(float intensity, int bands) {
    float bandSize = 1.0 / float(bands);
    float band = floor(intensity / bandSize);
    return band * bandSize;
}

// ============================================================================
// WATER FOAM
// ============================================================================

float calculateFoam(float depth, float time, vec2 worldPos) {
    const float foamThreshold = 0.8;  // Increased from 0.5 for more foam coverage
    float foamDiff = saturate(depth / foamThreshold);
    
    // Foam texture (noise) - FASTER animation
    float foamNoise = noise2D(worldPos * 4.0 + time * vec2(0.5, 0.4));  // Increased speed 2.5x
    
    // Animated foam lines (tutorial's sine wave technique) - FASTER movement
    float foamLines = sin((foamDiff - time * 0.8) * 8.0 * PI);  // Increased from 0.3 to 0.8
    foamLines = saturate(foamLines) * (1.0 - foamDiff);
    
    // Combine with more contrast
    float foam = step(foamDiff - foamLines * 0.5, foamNoise);  // Added multiplier for better visibility
    return foam * (1.0 - saturate(depth / foamThreshold));
}

// ============================================================================
// CLOUD RENDERING (Simplified for ShaderToy)
// ============================================================================

float getCloudDensity(vec3 p) {
    // ADJUSTED: Lower minimum height to prevent cutoff near horizon
    if (p.y < 1.0 || p.y > 5.0) return 0.0;  // Changed from 1.5 to 1.0
    
    // Simple cloud density with better distribution
    float density = fbm(p * 0.5 + vec3(iTime * 0.02, 0.0, 0.0), 4);
    
    // Smoother threshold for less cutoff
    density = smoothstep(0.35, 0.75, density);  // Changed from 0.4, 0.8
    
    // Fade out at bottom edge to prevent hard cutoff
    float bottomFade = smoothstep(1.0, 1.8, p.y);  // Smooth transition zone
    density *= bottomFade;
    
    return density * 0.5;
}

vec4 raymarchClouds(vec3 ro, vec3 rd) {
    vec3 col = vec3(0.0);
    float alpha = 0.0;
    
    float t = 2.0;
    const int maxSteps = 60;
    const float stepSize = 0.05;
    
    vec3 lightDir = normalize(vec3(0.7, 0.6, -0.3));
    
    for (int i = 0; i < maxSteps; i++) {
        if (t > 15.0 || alpha > 0.95) break;
        
        vec3 pos = ro + rd * t;
        float density = getCloudDensity(pos);
        
        if (density > 0.01) {
            // TOON SHADING FOR CLOUDS
            // Calculate simple lighting based on position
            float lightSample = getCloudDensity(pos + lightDir * 0.3);
            float shadow = 1.0 - lightSample;
            
            // Quantize lighting into discrete bands (toon style)
            shadow = toonShading(shadow, 3);  // 3 bands for clouds
            
            // Cloud base colors with toon shading
            vec3 cloudLight = vec3(1.0, 1.0, 1.0);      // Bright white
            vec3 cloudMid = vec3(0.85, 0.88, 0.92);     // Light gray
            vec3 cloudDark = vec3(0.65, 0.70, 0.78);    // Shadow gray
            
            // Apply banded lighting
            vec3 cloudCol;
            if (shadow > 0.66) {
                cloudCol = cloudLight;
            } else if (shadow > 0.33) {
                cloudCol = cloudMid;
            } else {
                cloudCol = cloudDark;
            }
            
            float stepAlpha = density * stepSize * 8.0;
            stepAlpha = min(stepAlpha, 1.0);
            
            col += cloudCol * stepAlpha * (1.0 - alpha);
            alpha += stepAlpha * (1.0 - alpha);
        }
        
        t += stepSize;
    }
    
    return vec4(col, alpha);
}

// ============================================================================
// WATER RENDERING WITH FRESNEL
// ============================================================================

vec3 renderWater(vec3 ro, vec3 rd, vec3 skyColor) {
    // Ray march to find water surface intersection with height displacement
    float t = 0.1;
    const int maxSteps = 100;
    const float epsilon = 0.01;
    bool hit = false;
    vec3 hitPos;
    
    // Ray march through the height field
    for (int i = 0; i < maxSteps; i++) {
        hitPos = ro + rd * t;
        
        // Get wave height at this XZ position
        float waveHeight = getWaveHeight(hitPos.xz, iTime);
        
        // Check if we're below the wave surface
        float surfaceY = waveHeight;  // Base water level = 0 + wave displacement
        float diff = hitPos.y - surfaceY;
        
        if (diff < epsilon && diff > -epsilon) {
            hit = true;
            break;
        }
        
        // Step forward (use diff as hint for step size)
        t += max(0.02, abs(diff) * 0.5);
        
        if (t > 50.0) break;  // Max distance
    }
    
    if (!hit) return vec3(0.0);  // No intersection
    
    // Calculate distance for LOD-based anti-aliasing
    float distance = t;
    
    // Get water normal with distance-based filtering
    // Calculate from actual height field for accurate lighting
    float h = 0.02;
    float heightCenter = getWaveHeight(hitPos.xz, iTime);
    float heightRight = getWaveHeight(hitPos.xz + vec2(h, 0.0), iTime);
    float heightUp = getWaveHeight(hitPos.xz + vec2(0.0, h), iTime);
    
    float dx = (heightRight - heightCenter) / h;
    float dz = (heightUp - heightCenter) / h;
    
    vec3 normal = normalize(vec3(-dx, 1.0, -dz));
    
    // Blend with procedural normal maps for additional detail
    vec3 detailNormal = getWaterNormal(hitPos.xz, iTime, distance);
    normal = normalize(normal + detailNormal * 0.3);  // 30% detail normal influence
    
    // Simulate depth (use distance as proxy)
    float depth = min(t * 0.2, 3.0);
    
    // ========================================================================
    // THREE-TIER COLOR SYSTEM (Tutorial approach) - ENHANCED for visibility
    // ========================================================================
    vec3 intersectionColor = vec3(0.7, 0.95, 1.0);   // Brighter cyan (shore)
    vec3 waterColor = vec3(0.35, 0.75, 0.98);        // Brighter mid blue
    vec3 fogColor = vec3(0.2, 0.55, 0.85);           // Lighter deep blue
    
    const float intersectionThreshold = 0.5;  // Increased for more gradient
    const float fogThreshold = 2.5;           // Increased for smoother transition
    
    float intersectionDiff = saturate(depth / intersectionThreshold);
    float fogDiff = saturate(depth / fogThreshold);
    
    // Smooth the transitions to reduce aliasing
    intersectionDiff = smoothstep(0.0, 1.0, intersectionDiff);
    fogDiff = smoothstep(0.0, 1.0, fogDiff);
    
    vec3 baseColor = mix(mix(intersectionColor, waterColor, intersectionDiff), 
                         fogColor, fogDiff);
    
    // ========================================================================
    // LIGHTING WITH TOON SHADING
    // ========================================================================
    vec3 lightDir = normalize(vec3(0.7, 0.6, -0.3));
    vec3 viewDir = -rd;
    
    // Diffuse (toon-shaded)
    float diffuse = max(0.0, dot(normal, lightDir));
    float toonDiffuse = toonShading(diffuse, 3);
    
    // Specular (toon-shaded) - INCREASED for more visible wave highlights
    vec3 halfVec = normalize(lightDir + viewDir);
    float specular = pow(max(0.0, dot(normal, halfVec)), 24.0);  // Reduced from 32 for wider highlights
    float toonSpecular = toonShading(specular, 2);
    
    // Combine lighting - BRIGHTER for more wave visibility
    vec3 ambient = baseColor * 0.7;  // Increased
    vec3 diffuseColor = baseColor * toonDiffuse * 0.9;  // Increased
    vec3 specularColor = vec3(1.0) * toonSpecular * 0.8;  // Increased significantly
    
    vec3 shadedColor = ambient + diffuseColor + specularColor;
    
    // ========================================================================
    // FRESNEL EFFECT (Mentioned in proposal!)
    // ========================================================================
    // Fresnel makes water more reflective at glancing angles
    float fresnel = pow(1.0 - max(0.0, dot(normal, viewDir)), 3.0);
    fresnel = toonShading(fresnel, 2);  // Quantize for stylized look
    
    // Mix with sky reflection based on Fresnel
    shadedColor = mix(shadedColor, skyColor, fresnel * 0.25);
    
    // ========================================================================
    // FOAM
    // ========================================================================
    float foam = calculateFoam(depth, iTime, hitPos.xz);
    vec3 foamColor = vec3(1.0);
    shadedColor = mix(shadedColor, foamColor, foam * 0.9);
    
    return shadedColor;
}

// ============================================================================
// MAIN IMAGE
// ============================================================================

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    // Normalized pixel coordinates (FIXED: negative y to correct orientation)
    vec2 uv = (fragCoord * 2.0 - iResolution.xy) / iResolution.y;
    
    // Camera setup
    vec3 ro = vec3(0.0, 1.5, -4.0);  // Camera position
    vec3 rd = normalize(vec3(uv.x, uv.y, 1.0));  // Ray direction (already correct after uv flip)
    
    // Sky gradient
    float skyGradient = smoothstep(-0.5, 0.8, rd.y);
    vec3 skyColor = mix(vec3(0.6, 0.8, 0.95), vec3(0.75, 0.9, 1.0), skyGradient);
    
    vec3 finalColor = skyColor;
    
    // Render clouds (upper portion) - ADJUSTED threshold to prevent cutoff
    if (rd.y > 0.0) {  // Changed from 0.2 to 0.0 to show more clouds
        vec4 cloudResult = raymarchClouds(ro, rd);
        finalColor = mix(finalColor, cloudResult.rgb, cloudResult.a);
    }
    
    // Render water (lower portion) - ADJUSTED for better coverage
    if (rd.y < 0.4) {  // Changed from 0.3 to 0.4 for smoother transition
        vec3 waterColor = renderWater(ro, rd, skyColor);
        if (dot(waterColor, waterColor) > 0.0) {
            finalColor = waterColor;
        }
    }
    
    fragColor = vec4(finalColor, 1.0);
}
