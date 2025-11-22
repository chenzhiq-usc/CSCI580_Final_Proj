#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>

// ============================================================================
// VECTOR3 CLASS - 3D Vector Math Utilities
// ============================================================================
struct Vec3 {
    float x, y, z;
    
    Vec3(float x = 0, float y = 0, float z = 0) : x(x), y(y), z(z) {}
    
    // Vector operations
    Vec3 operator+(const Vec3& v) const { return Vec3(x + v.x, y + v.y, z + v.z); }
    Vec3 operator-(const Vec3& v) const { return Vec3(x - v.x, y - v.y, z - v.z); }
    Vec3 operator*(float s) const { return Vec3(x * s, y * s, z * s); }
    Vec3 operator/(float s) const { return Vec3(x / s, y / s, z / s); }
    
    // Vector length and normalization
    float length() const { return std::sqrt(x * x + y * y + z * z); }
    
    Vec3 normalize() const {
        float len = length();
        return len > 0 ? Vec3(x / len, y / len, z / len) : Vec3(0, 0, 0);
    }
    
    // Dot product
    float dot(const Vec3& v) const { return x * v.x + y * v.y + z * v.z; }
    
    // Cross product
    Vec3 cross(const Vec3& v) const {
        return Vec3(
            y * v.z - z * v.y,
            z * v.x - x * v.z,
            x * v.y - y * v.x
        );
    }
};

// ============================================================================
// UTILITY FUNCTIONS - Math helpers for noise and interpolation
// ============================================================================

// Fractional part of a number
float fract(float x) { return x - std::floor(x); }

// Fractional part for vectors
Vec3 fract(const Vec3& v) {
    return Vec3(fract(v.x), fract(v.y), fract(v.z));
}

// Smooth interpolation function (cubic Hermite)
float smoothstep(float edge0, float edge1, float x) {
    float t = std::max(0.0f, std::min(1.0f, (x - edge0) / (edge1 - edge0)));
    return t * t * (3 - 2 * t);
}

// Linear interpolation
float mix(float a, float b, float t) {
    return a * (1 - t) + b * t;
}

// Clamp value between min and max
float clamp(float x, float minVal, float maxVal) {
    return std::max(minVal, std::min(maxVal, x));
}

// Saturate - clamp to 0-1
float saturate(float x) {
    return clamp(x, 0.0f, 1.0f);
}

// ============================================================================
// NOISE FUNCTIONS - Procedural noise generation for clouds and waves
// ============================================================================

// Hash function for pseudo-random 3D vectors
Vec3 hash3(const Vec3& p) {
    float x = std::sin(p.x * 443.897f + p.y * 441.423f + p.z * 437.195f) * 43758.5453f;
    float y = std::sin(p.x * 419.123f + p.y * 431.654f + p.z * 421.321f) * 43758.5453f;
    float z = std::sin(p.x * 433.789f + p.y * 427.456f + p.z * 439.654f) * 43758.5453f;
    return Vec3(fract(x), fract(y), fract(z));
}

// 2D hash function
float hash2D(float x, float y) {
    float result = std::sin(x * 127.1f + y * 311.7f) * 43758.5453123f;
    return fract(result);
}

// Perlin Noise - Smooth 3D noise function
float noise3D(const Vec3& p) {
    Vec3 i(std::floor(p.x), std::floor(p.y), std::floor(p.z));
    Vec3 f(p.x - i.x, p.y - i.y, p.z - i.z);
    Vec3 u(f.x * f.x * (3 - 2 * f.x), f.y * f.y * (3 - 2 * f.y), f.z * f.z * (3 - 2 * f.z));
    
    float result = 0;
    for (int z = 0; z <= 1; z++) {
        for (int y = 0; y <= 1; y++) {
            for (int x = 0; x <= 1; x++) {
                Vec3 corner(i.x + x, i.y + y, i.z + z);
                Vec3 h = hash3(corner);
                float wx = (x == 0) ? (1 - u.x) : u.x;
                float wy = (y == 0) ? (1 - u.y) : u.y;
                float wz = (z == 0) ? (1 - u.z) : u.z;
                result += (h.x - 0.5f) * wx * wy * wz; 
            }
        }
    }
    return result * 0.5f + 0.5f; 
}

// 2D Perlin noise (for water surface)
float noise2D(float x, float y) {
    float ix = std::floor(x);
    float iy = std::floor(y);
    float fx = x - ix;
    float fy = y - iy;
    
    float ux = fx * fx * (3.0f - 2.0f * fx);
    float uy = fy * fy * (3.0f - 2.0f * fy);
    
    float a = hash2D(ix, iy);
    float b = hash2D(ix + 1.0f, iy);
    float c = hash2D(ix, iy + 1.0f);
    float d = hash2D(ix + 1.0f, iy + 1.0f);
    
    return mix(mix(a, b, ux), mix(c, d, ux), uy);
}

// Fractal Brownian Motion - Layered noise for detail
float fbm(const Vec3& p, int octaves) {
    float total = 0.0f;
    float amplitude = 0.5f;      // Initial strength of each octave
    float frequency = 1.0f;      // Initial frequency
    float gain = 0.5f;           // How much amplitude decreases each octave
    float lacunarity = 2.0f;     // How much frequency increases each octave
    
    for (int i = 0; i < octaves; i++) {
        total += noise3D(p * frequency) * amplitude; 
        frequency *= lacunarity;
        amplitude *= gain;
    }
    return total;
}

// Worley Noise - Cellular/voronoi noise for cloud shapes
float worley3D(const Vec3& p) {
    Vec3 id(std::floor(p.x), std::floor(p.y), std::floor(p.z));
    Vec3 fd(p.x - id.x, p.y - id.y, p.z - id.z);
    
    float minDistSq = 1.0f; 
    
    // Check neighboring cells for closest point
    for (int z = -1; z <= 1; z++) {
        for (int y = -1; y <= 1; y++) {
            for (int x = -1; x <= 1; x++) {
                Vec3 coord(id.x + x, id.y + y, id.z + z);
                Vec3 pointPos = hash3(coord);
                Vec3 diff(x + pointPos.x - fd.x, y + pointPos.y - fd.y, z + pointPos.z - fd.z);
                
                float distSq = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
                minDistSq = std::min(minDistSq, distSq);
            }
        }
    }
    return minDistSq;
}

// ============================================================================
// CLOUD SYSTEM - Volumetric cloud rendering
// ============================================================================

// Calculate density for a single cloud cluster
float singleCloudDensity(Vec3 p, Vec3 center, float size) {
    // Distance Falloff - Creates spherical base shape
    Vec3 offset = p - center;
    float dist = offset.length();
    float falloff = smoothstep(size * 1.5f, size * 0.4f, dist); 
    
    if (falloff < 0.001f) return 0.0f;
    
    // Make shape more spherical/rounded
    falloff = std::pow(falloff, 1.2f); 

    // Base cloud shape using Worley noise
    float baseFreq = 2.0f / size; 
    float worleyShape = worley3D(p * (baseFreq * 0.3f)); 
    
    // Convert distance to density (closer = higher density)
    float baseDensity = 1.0f - std::pow(worleyShape, 4.0f); 
    
    // Add detail with fractal noise
    float perlinFBM = fbm(p * baseFreq * 1.8f, 5); 
    
    // Combine base shape with details
    float totalDensityBase = baseDensity * perlinFBM; 
    totalDensityBase = std::min(totalDensityBase, 1.0f);    
    totalDensityBase *= 3.5f; // Density multiplier
    totalDensityBase *= falloff; // Apply boundary falloff
    
    // Create soft, marshmallow-like edges
    float density = smoothstep(0.5f, 1.5f, totalDensityBase); 
    
    return density;
}

// Cloud cluster data structure
struct CloudCluster {
    Vec3 center;
    float size;
    Vec3 velocity;
};

// Random number generator
std::default_random_engine generator(std::chrono::system_clock::now().time_since_epoch().count());

// Create randomized cloud positions
std::vector<CloudCluster> createCloudClusters() {
    std::vector<CloudCluster> clusters;
    const int numClusters = 18;

    // Random distribution ranges
    std::uniform_real_distribution<float> distX(-5.0f, 5.0f);   // Horizontal spread
    std::uniform_real_distribution<float> distY(2.0f, 3.5f);    // Height variation (HIGHER in sky)
    std::uniform_real_distribution<float> distZ(1.0f, 15.0f);   // Depth range
    std::uniform_real_distribution<float> distSize(0.8f, 1.5f); // Size variation
    std::uniform_real_distribution<float> distSpeed(0.01f, 0.045f); // Movement speed

    for (int i = 0; i < numClusters; ++i) {
        float cx = distX(generator);
        float cy = distY(generator);         
        float cz = distZ(generator);
        float size = distSize(generator);
        float speed = distSpeed(generator);
        
        // Clouds move to the right (positive X)
        Vec3 velocity = {speed, 0.0f, 0.0f};

        clusters.push_back({
            {cx, cy, cz}, 
            size, 
            velocity
        });
    }
    
    return clusters;
}

// Get combined density from all cloud clusters
float getCloudDensity(Vec3 p, float time) {
    static std::vector<CloudCluster> clusters = createCloudClusters();
    
    float totalDensity = 0.0f;
    
    // Clouds are in the upper part of the scene (y > 2.0)
    if (p.y < 1.5f || p.y > 5.0f) {
        return 0.0f;
    }

    // Sample each cloud cluster
    for (const auto& cluster : clusters) {
        Vec3 animatedCenter = cluster.center + cluster.velocity * time;
        
        // Wrap clouds around horizontally
        if (animatedCenter.x > 4.0f) {
            animatedCenter.x = std::fmod(animatedCenter.x + 4.0f, 8.0f) - 4.0f;
        }
        
        float density = singleCloudDensity(p, animatedCenter, cluster.size);
        totalDensity += density * 0.6f; // Reduce contribution to prevent over-saturation
    }
    
    totalDensity = std::min(totalDensity, 1.0f);
    
    return totalDensity;
}

// Beer's Law - Light attenuation through medium
float beerLaw(float density, float distance) {
    return std::exp(-density * distance * 3.0f);
}

// Light marching through clouds for shadows
float lightMarch(Vec3 pos, const Vec3& lightDir, float time) {
    float totalDensity = 0.0f;
    const float marchSize = 0.03f; 
    const int steps = 25; 
    
    for (int i = 0; i < steps; i++) {
        pos = pos + lightDir * marchSize;
        float density = getCloudDensity(pos, time);
        totalDensity += density * marchSize;
        
        if (totalDensity > 5.0f) break; 
    }
    
    return beerLaw(totalDensity, 1.0f);
}

// ============================================================================
// WAVE HEIGHT DISPLACEMENT - Creates actual up/down waves
// ============================================================================

float getWaveHeight(float x, float z, float time) {
    // Multiple layers of noise for wave height
    float height = 0.0f;
    
    // Large waves (primary swell)
    height += noise2D(x * 0.8f + time * 0.15f, z * 0.8f + time * 0.10f) * 0.15f;
    
    // Medium waves
    height += noise2D(x * 1.2f - time * 0.12f, z * 1.2f + time * 0.15f) * 0.10f;
    
    // Small ripples
    height += noise2D(x * 2.0f + time * 0.08f, z * 2.0f - time * 0.08f) * 0.05f;
    
    // Bias to center around 0
    height = (height - 0.15f) * 2.0f;
    
    return height;
}

// ============================================================================
// STYLIZED WATER SYSTEM - Based on normal maps, not vertex displacement
// ============================================================================

// Simple 2D normal map simulation using noise
Vec3 sampleWaterNormal(float x, float z, float time, float scale, Vec3 panSpeed) {
    // Simulate normal map sampling with animated noise
    float offsetX = time * panSpeed.x;
    float offsetZ = time * panSpeed.z;
    
    float u = x * scale + offsetX;
    float v = z * scale + offsetZ;
    
    // Sample noise to create normal variations
    float h = 0.01f;
    float heightCenter = noise2D(u, v);
    float heightRight = noise2D(u + h, v);
    float heightUp = noise2D(u, v + h);
    
    // Calculate normal using finite differences
    float dx = (heightRight - heightCenter) / h;
    float dz = (heightUp - heightCenter) / h;
    
    Vec3 normal(-dx * 0.3f, 1.0f, -dz * 0.3f); // 0.3f controls normal strength
    return normal.normalize();
}

// Blend two normal maps (additive blending like in tutorial)
Vec3 blendNormals(const Vec3& n1, const Vec3& n2) {
    return Vec3(n1.x + n2.x, n1.y * n2.y, n1.z + n2.z).normalize();
}

// Get combined water normal at a position
Vec3 getWaterNormal(float x, float z, float time) {
    // Two normal maps with different scales and panning speeds
    // (simulating the tutorial's _NormalA and _NormalB)
    
    Vec3 normalA = sampleWaterNormal(x, z, time, 1.0f, Vec3(0.05f, 0.0f, 0.03f));
    Vec3 normalB = sampleWaterNormal(x, z, time, 1.5f, Vec3(-0.03f, 0.0f, 0.04f));
    
    return blendNormals(normalA, normalB);
}

// Water foam calculation (shoreline and depth-based)
float calculateWaterFoam(float depth, float time, float x, float z) {
    // Foam appears at shallow depths
    const float foamThreshold = 0.5f; // How far from shore foam appears
    
    float foamDiff = saturate(depth / foamThreshold);
    
    // Foam texture simulation (noise-based)
    float foamNoise = noise2D(x * 3.0f + time * 0.2f, z * 3.0f + time * 0.15f);
    
    // Animated foam lines moving inward (from tutorial's sine wave effect)
    const float PI = 3.14159265f;
    float foamLines = std::sin((foamDiff - time * 0.3f) * 8.0f * PI);
    foamLines = saturate(foamLines) * (1.0f - foamDiff); // Restrict to shore area
    
    // Combine foam elements
    float foam = (foamDiff - foamLines < foamNoise) ? 1.0f : 0.0f;
    
    return foam * (1.0f - saturate(depth / foamThreshold));
}

// ============================================================================
// TOON/CEL SHADING - Stylized lighting for cartoon aesthetic
// ============================================================================

// Quantize lighting into discrete bands for toon shading
float toonShading(float intensity, int bands) {
    float bandSize = 1.0f / float(bands);
    float band = std::floor(intensity / bandSize);
    return band * bandSize;
}

// ============================================================================
// WATER RENDERING - Stylized water with depth-based colors
// ============================================================================

struct OceanHit {
    bool hit;
    Vec3 position;
    Vec3 normal;
    float distance;
    float depth; // Depth below surface
};

// Raymarch to find ocean surface intersection with height displacement
OceanHit rayMarchOcean(const Vec3& ro, const Vec3& rd, float time, float maxDist) {
    OceanHit result;
    result.hit = false;
    
    float t = 0.1f;
    const int maxSteps = 50;  // Reduced from 100 for 2x speed
    const float epsilon = 0.02f;  // Slightly relaxed tolerance for speed
    
    for (int i = 0; i < maxSteps && t < maxDist; i++) {
        Vec3 pos = ro + rd * t;
        
        // Get wave height at current XZ position
        float waveHeight = getWaveHeight(pos.x, pos.z, time);
        float surfaceY = waveHeight;  // Surface at wave height
        
        // Check if we're at the surface
        float dist = pos.y - surfaceY;
        
        // Hit detection
        if (std::abs(dist) < epsilon) {
            result.hit = true;
            result.position = pos;
            result.distance = t;
            
            // Calculate normal from height field using finite differences
            float h = 0.03f;  // Slightly larger for speed
            float heightCenter = waveHeight;
            float heightRight = getWaveHeight(pos.x + h, pos.z, time);
            float heightUp = getWaveHeight(pos.x, pos.z + h, time);
            
            float dx = (heightRight - heightCenter) / h;
            float dz = (heightUp - heightCenter) / h;
            
            result.normal = Vec3(-dx, 1.0f, -dz).normalize();
            
            // Blend with detail normals for additional texture
            Vec3 detailNormal = getWaterNormal(pos.x, pos.z, time);
            result.normal = (result.normal + detailNormal * 0.3f).normalize();
            
            break;
        }
        
        // Adaptive step size (larger when far from surface)
        t += std::max(0.03f, std::abs(dist) * 0.5f);  // Increased minimum step
    }
    
    return result;
}

// Render stylized water with toon shading
Vec3 renderWater(const Vec3& ro, const Vec3& rd, float time, const Vec3& skyColor) {
    OceanHit hit = rayMarchOcean(ro, rd, time, 50.0f);
    
    if (!hit.hit) {
        return Vec3(0, 0, 0); // No hit - return black (will use sky instead)
    }
    
    // Calculate depth for color and foam
    hit.depth = std::min(hit.distance * 0.2f, 3.0f);
    
    // Water colors from tutorial approach (three-level system)
    // Match GLSL brighter colors
    Vec3 intersectionColor(0.7f, 0.95f, 1.0f);   // Brighter cyan (shore)
    Vec3 waterColor(0.35f, 0.75f, 0.98f);        // Brighter mid blue
    Vec3 fogColor(0.2f, 0.55f, 0.85f);           // Lighter deep blue
    
    // Depth-based color interpolation (from tutorial)
    const float intersectionThreshold = 0.3f;
    const float fogThreshold = 2.0f;
    
    float intersectionDiff = saturate(hit.depth / intersectionThreshold);
    float fogDiff = saturate(hit.depth / fogThreshold);
    
    // Three-tier color system
    Vec3 baseColor = Vec3(
        mix(mix(intersectionColor.x, waterColor.x, intersectionDiff), fogColor.x, fogDiff),
        mix(mix(intersectionColor.y, waterColor.y, intersectionDiff), fogColor.y, fogDiff),
        mix(mix(intersectionColor.z, waterColor.z, intersectionDiff), fogColor.z, fogDiff)
    );
    
    // Lighting setup
    Vec3 lightDir = Vec3(0.7f, 0.6f, -0.3f).normalize();
    Vec3 viewDir = (rd * -1.0f).normalize();
    
    // Diffuse lighting with toon quantization
    float diffuse = std::max(0.0f, hit.normal.dot(lightDir));
    float toonDiffuse = toonShading(diffuse, 3); // 3 bands like tutorial
    
    // Specular highlight (Blinn-Phong with toon quantization)
    Vec3 halfVec = (lightDir + viewDir).normalize();
    float specular = std::pow(std::max(0.0f, hit.normal.dot(halfVec)), 32.0f);
    float toonSpecular = toonShading(specular, 2); // 2 bands
    
    // Apply lighting - match GLSL brightness
    Vec3 ambient = baseColor * 0.7f;
    Vec3 diffuseColor = baseColor * toonDiffuse * 0.9f;
    Vec3 specularColor = Vec3(1.0f, 1.0f, 1.0f) * toonSpecular * 0.8f;
    
    Vec3 shadedColor = ambient + diffuseColor + specularColor;
    
    // Add foam
    float foam = calculateWaterFoam(hit.depth, time, hit.position.x, hit.position.z);
    Vec3 foamColor(1.0f, 1.0f, 1.0f);
    shadedColor = Vec3(
        mix(shadedColor.x, foamColor.x, foam * 0.9f),
        mix(shadedColor.y, foamColor.y, foam * 0.9f),
        mix(shadedColor.z, foamColor.z, foam * 0.9f)
    );
    
    // Fresnel effect for transparency/reflectivity
    float fresnel = std::pow(1.0f - std::max(0.0f, hit.normal.dot(viewDir)), 3.0f);
    fresnel = toonShading(fresnel, 2); // Quantize fresnel
    
    // Mix with sky reflection
    shadedColor = Vec3(
        mix(shadedColor.x, skyColor.x, fresnel * 0.25f),
        mix(shadedColor.y, skyColor.y, fresnel * 0.25f),
        mix(shadedColor.z, skyColor.z, fresnel * 0.25f)
    );
    
    return shadedColor;
}

// ============================================================================
// CLOUD RENDERING - Volumetric ray marching
// ============================================================================

struct CloudResult {
    Vec3 color;
    float alpha;
};

CloudResult raymarchClouds(const Vec3& ro, const Vec3& rd, float time) {
    Vec3 col(0, 0, 0);
    float alpha = 0.0f;
    
    float t = 2.0f; 
    
    Vec3 lightDir = Vec3(0.7f, 0.6f, -0.3f).normalize();
    Vec3 lightColor(1.0f, 0.98f, 0.9f);
    
    const int maxSteps = 100;  // Reduced from 250 for speed
    const float stepSize = 0.02f;  // Increased from 0.01 for speed
    const float scatterCoeff = 12.0f; 
    
    for (int i = 0; i < maxSteps; i++) {
        Vec3 pos = ro + rd * t;
        
        if (t > 15.0f || alpha > 0.98f) {
            break;
        }
        
        float density = getCloudDensity(pos, time);
        
        if (density > 0.001f) {
            float lightEnergy = lightMarch(pos, lightDir, time);
            
            // Ambient and direct lighting
            Vec3 ambient(0.7f, 0.75f, 0.85f);
            ambient = ambient * 0.6f;
            Vec3 direct = lightColor * (lightEnergy * 0.5f);
            
            Vec3 cloudCol(
                ambient.x + direct.x,
                ambient.y + direct.y,
                ambient.z + direct.z
            );
            
            float stepAlpha = density * stepSize * scatterCoeff; 
            stepAlpha = std::min(stepAlpha, 1.0f);
            
            col.x += cloudCol.x * stepAlpha * (1 - alpha);
            col.y += cloudCol.y * stepAlpha * (1 - alpha);
            col.z += cloudCol.z * stepAlpha * (1 - alpha);
            alpha += stepAlpha * (1 - alpha);
        }
        
        t += stepSize;
    }
    
    return {col, alpha};
}

// ============================================================================
// MAIN RENDERING FUNCTION - Combines ocean, clouds, and sky
// ============================================================================

void render(int width, int height, float time, std::vector<unsigned char>& pixels) {
    std::cout << "Rendering frame at time: " << time << "s" << std::endl;
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // Convert pixel coordinates to normalized device coordinates
            float px = (x * 2.0f - width) / float(height);
            float py = (y * 2.0f - height) / float(height);
            
            // Camera setup - FIXED ORIENTATION
            Vec3 ro(0, 1.5f, -4.0f);  // Camera position (eye point) - looking forward
            Vec3 rd = Vec3(px, -py, 1.0f).normalize(); // Ray direction - NEGATIVE py fixes flip
            
            Vec3 finalColor;
            
            // Sky gradient (based on ray direction) - brighter, more vibrant
            float skyGradient = smoothstep(-0.5f, 0.8f, rd.y);
            Vec3 skyColor(
                mix(0.6f, 0.75f, skyGradient),   // Lighter horizon
                mix(0.8f, 0.9f, skyGradient),    // More vibrant
                mix(0.95f, 1.0f, skyGradient)    // Brighter blue
            );
            
            // Start with sky
            finalColor = skyColor;
            
            // Render clouds first (only in upper portion where rd.y > 0.1)
            if (rd.y > 0.1f) {
                CloudResult cloudResult = raymarchClouds(ro, rd, time);
                finalColor = Vec3(
                    mix(finalColor.x, cloudResult.color.x, cloudResult.alpha),
                    mix(finalColor.y, cloudResult.color.y, cloudResult.alpha),
                    mix(finalColor.z, cloudResult.color.z, cloudResult.alpha)
                );
            }
            
            // Render water only if looking down (if ray points downward)
            // Don't render water over clouds!
            if (rd.y < 0.1f) {  // Changed from 0.3f to prevent overlap with clouds
                Vec3 waterColor = renderWater(ro, rd, time, skyColor);
                
                // If water was hit, use water color
                if (waterColor.x > 0.0f || waterColor.y > 0.0f || waterColor.z > 0.0f) {
                    finalColor = waterColor;
                }
            }
            
            // Write pixel to buffer
            int idx = (y * width + x) * 3;
            pixels[idx + 0] = static_cast<unsigned char>(clamp(finalColor.x * 255, 0.0f, 255.0f));
            pixels[idx + 1] = static_cast<unsigned char>(clamp(finalColor.y * 255, 0.0f, 255.0f));
            pixels[idx + 2] = static_cast<unsigned char>(clamp(finalColor.z * 255, 0.0f, 255.0f));
        }
        
        // Progress indicator
        if (y % 10 == 0) {
            std::cout << "Render Progress: " << (y * 100 / height) << "%\r" << std::flush;
        }
    }
    std::cout << "Render Progress: 100%   " << std::endl;
}

// ============================================================================
// FILE I/O - Save rendered image as PPM
// ============================================================================

void savePPM(const std::string& filename, int width, int height, const std::vector<unsigned char>& pixels) {
    std::ofstream file(filename, std::ios::binary);
    file << "P6\n" << width << " " << height << "\n255\n";
    file.write(reinterpret_cast<const char*>(pixels.data()), pixels.size());
    file.close();
    std::cout << "Image saved to: " << filename << std::endl;
}

// ============================================================================
// MAIN - Program entry point
// ============================================================================

int main() {
    // Render settings
    const int width = 800;           // Image width (reduced from 1920)
    const int height = 600;          // Image height (reduced from 1080)
    const int numFrames = 60;        // Number of frames (reduced from 120 for 1 second @ 30fps)
    const float fps = 30.0f;         // Frames per second
    const float frameDuration = 1.0f / fps;
    
    std::cout << "=====================================" << std::endl;
    std::cout << "  Stylized Ocean Scene Renderer" << std::endl;
    std::cout << "  (Fixed orientation + gentle waves)" << std::endl;
    std::cout << "=====================================" << std::endl;
    std::cout << "Resolution: " << width << "x" << height << std::endl;
    std::cout << "Frames: " << numFrames << " @ " << fps << " FPS" << std::endl;
    std::cout << "Features: Stylized water + Clouds" << std::endl;
    std::cout << "=====================================" << std::endl;
    
    // Render each frame
    for (int frame = 0; frame < numFrames; frame++) {
        std::vector<unsigned char> pixels(width * height * 3);
        float time = frame * frameDuration;
        
        std::cout << "\n[Frame " << (frame + 1) << "/" << numFrames << "]" << std::endl;
        render(width, height, time, pixels);
        
        // Save frame to output directory
        char filename[256];
        snprintf(filename, sizeof(filename), "output/ocean_frame_%04d.ppm", frame);
        savePPM(filename, width, height, pixels);
    }
    
    std::cout << "\n=====================================" << std::endl;
    std::cout << "  Rendering Complete!" << std::endl;
    std::cout << "=====================================" << std::endl;
    std::cout << "To convert to PNG:" << std::endl;
    std::cout << "convert ocean_frame_0000.ppm ocean_frame_0000.png" << std::endl;
    
    return 0;
}
