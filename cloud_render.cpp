#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>

// ============ Vector3 Class ============
struct Vec3 {
    float x, y, z;
    
    Vec3(float x = 0, float y = 0, float z = 0) : x(x), y(y), z(z) {}
    
    Vec3 operator+(const Vec3& v) const { return Vec3(x + v.x, y + v.y, z + v.z); }
    Vec3 operator-(const Vec3& v) const { return Vec3(x - v.x, y - v.y, z - v.z); }
    Vec3 operator*(float s) const { return Vec3(x * s, y * s, z * s); }
    
    float length() const { return std::sqrt(x * x + y * y + z * z); }
    
    Vec3 normalize() const {
        float len = length();
        return len > 0 ? Vec3(x / len, y / len, z / len) : Vec3(0, 0, 0);
    }
    
    float dot(const Vec3& v) const { return x * v.x + y * v.y + v.z * v.z; }
};

// ============ Utility Functions ============
float fract(float x) { return x - std::floor(x); }

Vec3 fract(const Vec3& v) {
    return Vec3(fract(v.x), fract(v.y), fract(v.z));
}

float smoothstep(float edge0, float edge1, float x) {
    float t = std::max(0.0f, std::min(1.0f, (x - edge0) / (edge1 - edge0)));
    return t * t * (3 - 2 * t);
}

float mix(float a, float b, float t) {
    return a * (1 - t) + b * t;
}

// ============ Noise Functions ============
Vec3 hash3(const Vec3& p) {
    float x = std::sin(p.x * 443.897f + p.y * 441.423f + p.z * 437.195f) * 43758.5453f;
    float y = std::sin(p.x * 419.123f + p.y * 431.654f + p.z * 421.321f) * 43758.5453f;
    float z = std::sin(p.x * 433.789f + p.y * 427.456f + p.z * 439.654f) * 43758.5453f;
    return Vec3(fract(x), fract(y), fract(z));
}

// Perlin Noise
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

// Fractal Brownian Motion (FBM)
float fbm(const Vec3& p, int octaves) {
    float total = 0.0f;
    float amplitude = 0.5f; 
    float frequency = 1.0f; 
    float gain = 0.5f;      
    float lacunarity = 2.0f; 
    
    for (int i = 0; i < octaves; i++) {
        total += noise3D(p * frequency) * amplitude; 
        frequency *= lacunarity;
        amplitude *= gain;
    }
    return total;
}

// Worley Noise
float worley3D(const Vec3& p) {
    Vec3 id(std::floor(p.x), std::floor(p.y), std::floor(p.z));
    Vec3 fd(p.x - id.x, p.y - id.y, p.z - id.z);
    
    float minDistSq = 1.0f; 
    
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

// ============ Single Cloud Density ============
float singleCloudDensity(Vec3 p, Vec3 center, float size) {
    // Distance Falloff (Spherical base for overall shape)
    Vec3 offset = p - center;
    float dist = offset.length();
    float falloff = smoothstep(size * 1.5f, size * 0.4f, dist); 
    
    if (falloff < 0.001f) return 0.0f;
    
    // Power operation to make the shape more spherical (rounder)
    falloff = std::pow(falloff, 1.2f); 

    // Base Shape
    float baseFreq = 2.0f / size; 
    
    // Use Worley noise (distance) to define the rounded base shape
    float worleyShape = worley3D(p * (baseFreq * 0.3f)); 
    
    // Convert distance to density. Closer (smaller value) means higher density
    // Power function creates soft boundaries.
    float baseDensity = 1.0f - std::pow(worleyShape, 4.0f); 
    
    // FBM Detail Noise (Mid-to-high frequency)
    float perlinFBM = fbm(p * baseFreq * 1.8f, 5); 
    
    // Combine Worley base shape with FBM details
    float totalDensityBase = baseDensity * perlinFBM; 
    totalDensityBase = std::min(totalDensityBase, 1.0f);    
    // density multiplier
    totalDensityBase *= 3.5f; 
    // boundary falloff
    totalDensityBase *= falloff; 
    
    // Use wide smoothstep to ensure soft, marshmallow-like edges
    float density = smoothstep(0.5f, 1.5f, totalDensityBase); 
    
    return density;
}

// ============ Cloud Cluster Definition ============
struct CloudCluster {
    Vec3 center;
    float size;
    Vec3 velocity;
};

// Random number generator setup
std::default_random_engine generator(std::chrono::system_clock::now().time_since_epoch().count());

// Create multiple cloud clusters (18 randomized positions)
std::vector<CloudCluster> createCloudClusters() {
    std::vector<CloudCluster> clusters;
    const int numClusters = 18;

    // Random distribution range
    std::uniform_real_distribution<float> distX(-5.0f, 5.0f);   // X-axis (horizontal)
    std::uniform_real_distribution<float> distY(0.3f, 1.2f);    // Y-axis (vertical height)
    std::uniform_real_distribution<float> distZ(1.0f, 15.0f);   // Z-axis (depth)
    
    // Random size and speed ranges
    std::uniform_real_distribution<float> distSize(0.8f, 1.5f);
    std::uniform_real_distribution<float> distSpeed(0.01f, 0.045f);

    for (int i = 0; i < numClusters; ++i) {
        float cx = distX(generator);
        float cy = distY(generator);         
        float cz = distZ(generator);
        
        float size = distSize(generator);
        float speed = distSpeed(generator);
        
        // Make sure velocity is always to the right (positive X direction)
        Vec3 velocity = {speed, 0.0f, 0.0f};

        clusters.push_back({
            {cx, cy, cz}, 
            size, 
            velocity
        });
    }
    
    return clusters;
}

// ============ Combined Cloud Density ============
float getCloudDensity(Vec3 p, float time) {
    static std::vector<CloudCluster> clusters = createCloudClusters();
    
    float totalDensity = 0.0f;

    // Y-axis flip to make the base round and the top sparse
    // Flip point Y_center = 1.6f / 2 = 0.8f
    p.y = 1.6f - p.y; 
    
    // Safety boundary
    if (p.y < 0.0f || p.y > 2.5f) {
        return 0.0f;
    }

    for (const auto& cluster : clusters) {
        Vec3 animatedCenter = cluster.center + cluster.velocity * time;
        
        if (animatedCenter.x > 4.0f) {
            animatedCenter.x = std::fmod(animatedCenter.x + 4.0f, 8.0f) - 4.0f;
        }
        
        Vec3 p_flipped_center = p;
        p_flipped_center.y = 1.6f - p_flipped_center.y;
        
        float density = singleCloudDensity(p_flipped_center, animatedCenter, cluster.size);
        
        // Reduce density contribution (0.6f) to prevent over combining
        totalDensity += density * 0.6f; 
    }
    
    totalDensity = std::min(totalDensity, 1.0f);
    
    // Bottom safety falloff
    float floorFade = smoothstep(0.0f, 0.2f, p.y);
    totalDensity *= floorFade;
    
    return totalDensity;
}

// Beer's Law
float beerLaw(float density, float distance) {
    return std::exp(-density * distance * 3.0f);
}

// ============ Light Marching ============
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

// ============ Raymarch Main Function ============
struct RaymarchResult {
    Vec3 color;
    float alpha;
    int samplesHit; 
};

RaymarchResult raymarchClouds(const Vec3& ro, const Vec3& rd, float time) {
    Vec3 col(0, 0, 0);
    float alpha = 0.0f;
    
    float t = 2.0f; 
    int samplesHit = 0;
    
    Vec3 lightDir = Vec3(0.7f, 0.5f, -0.2f).normalize();
    Vec3 lightColor(1.0f, 0.98f, 0.9f);
    
    // High smoothness, low scattering
    const int maxSteps = 250; 
    const float stepSize = 0.01f; 
    const float scatterCoeff = 12.0f; 
    
    for (int i = 0; i < maxSteps; i++) {
        Vec3 pos = ro + rd * t;
        
        // Max Raymarch distance T=15.0f for distant clouds at Z=15.0f
        if (t > 15.0f || alpha > 0.98f) {
            break;
        }
        
        float density = getCloudDensity(pos, time);
        
        if (density > 0.001f) {
            samplesHit++;
            
            float lightEnergy = lightMarch(pos, lightDir, time);
            
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
    
    return {col, alpha, samplesHit};
}

// ============ Render Function ============
void render(int width, int height, float time, std::vector<unsigned char>& pixels) {
    int cloudPixelCount = 0; 
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float px = (x * 2.0f - width) / float(height);
            float py = (y * 2.0f - height) / float(height);
            
            Vec3 ro(0, 0.4f, -2.5f); 
            
            // Expanded Field of View (FOV)
            Vec3 rd = Vec3(px, py + 0.2f, 0.7f).normalize(); 
            
            Vec3 finalColor;
            
            if (y < height / 2) {
                float skyGradient = smoothstep(-0.5f, 0.5f, rd.y);
                Vec3 skyColor(
                    mix(0.4f, 0.5f, skyGradient),
                    mix(0.6f, 0.75f, skyGradient),
                    mix(0.85f, 0.95f, skyGradient)
                );
                
                RaymarchResult result = raymarchClouds(ro, rd, time);
                
                if (result.alpha > 0.1f) {
                    cloudPixelCount++;
                }
                
                finalColor = Vec3(
                    mix(skyColor.x, result.color.x, result.alpha),
                    mix(skyColor.y, result.color.y, result.alpha),
                    mix(skyColor.z, result.color.z, result.alpha)
                );
            } 
            else {
                finalColor = Vec3(0.78f, 0.86f, 0.94f);
            }
            
            int idx = (y * width + x) * 3;
            pixels[idx + 0] = static_cast<unsigned char>(std::min(255.0f, finalColor.x * 255));
            pixels[idx + 1] = static_cast<unsigned char>(std::min(255.0f, finalColor.y * 255));
            pixels[idx + 2] = static_cast<unsigned char>(std::min(255.0f, finalColor.z * 255));
        }
        
        if (y % 10 == 0) {
            std::cout << "Render Progress: " << (y * 100 / height) << "%\r" << std::flush;
        }
    }
    std::cout << "Render Progress: 100%   " << std::endl;
}

// ============ PPM File Output ============
void savePPM(const std::string& filename, int width, int height, const std::vector<unsigned char>& pixels) {
    std::ofstream file(filename, std::ios::binary);
    file << "P6\n" << width << " " << height << "\n255\n";
    file.write(reinterpret_cast<const char*>(pixels.data()), pixels.size());
    file.close();
    std::cout << "Image saved to: " << filename << std::endl;
}

// ============ Main Function (Render Multiple Frames) ============
int main() {
    const int width = 800;
    const int height = 600;
    const int numFrames = 1; 
    
    std::cout << "Starting volume cloud animation rendering..." << std::endl;
    std::cout << "Resolution: " << width << "x" << height << std::endl;
    std::cout << "Number of frames: " << numFrames << std::endl;
    
    for (int frame = 0; frame < numFrames; frame++) {
        std::vector<unsigned char> pixels(width * height * 3);
        float time = frame * 0.15f; 
        
        std::cout << "\nRendering Frame " << (frame + 1) << "/" << numFrames << "..." << std::endl;
        render(width, height, time, pixels);
        
        char filename[256];
        sprintf(filename, "cloud_frame_%04d.ppm", frame);
        savePPM(filename, width, height, pixels);
    }
    
    std::cout << "\nAll frames rendered!" << std::endl;
    
    return 0;
}
