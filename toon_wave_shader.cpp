#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>

// ============================================================================
// MATHEMATICAL CONSTANTS AND STRUCTURES
// ============================================================================

const float PI = 3.14159265359f;
const float DRAG_MULT = 0.28f;

struct vec2
{
    float x, y;
    vec2(float x = 0, float y = 0) : x(x), y(y) {}
    vec2 operator+(const vec2 &v) const { return vec2(x + v.x, y + v.y); }
    vec2 operator-(const vec2 &v) const { return vec2(x - v.x, y - v.y); }
    vec2 operator*(float s) const { return vec2(x * s, y * s); }
    vec2 operator*(const vec2 &v) const { return vec2(x * v.x, y * v.y); }
    vec2 operator/(float s) const { return vec2(x / s, y / s); }
    void operator+=(const vec2 &v)
    {
        x += v.x;
        y += v.y;
    }
    void operator-=(const vec2 &v)
    {
        x -= v.x;
        y -= v.y;
    }
};

// Allow scalar * vec2
inline vec2 operator*(float s, const vec2 &v)
{
    return vec2(v.x * s, v.y * s);
}

struct vec3
{
    float x, y, z;
    vec3(float x = 0, float y = 0, float z = 0) : x(x), y(y), z(z) {}
    vec3 operator+(const vec3 &v) const { return vec3(x + v.x, y + v.y, z + v.z); }
    vec3 operator-(const vec3 &v) const { return vec3(x - v.x, y - v.y, z - v.z); }
    vec3 operator*(float s) const { return vec3(x * s, y * s, z * s); }
    vec3 operator/(float s) const { return vec3(x / s, y / s, z / s); }
    vec3 operator*(const vec3 &v) const { return vec3(x * v.x, y * v.y, z * v.z); }
    void operator+=(const vec3 &v)
    {
        x += v.x;
        y += v.y;
        z += v.z;
    }
};

// Allow scalar * vec3
inline vec3 operator*(float s, const vec3 &v)
{
    return vec3(v.x * s, v.y * s, v.z * s);
}

struct vec4
{
    float x, y, z, w;
    vec4(float x = 0, float y = 0, float z = 0, float w = 0) : x(x), y(y), z(z), w(w) {}
};

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

float saturate(float x)
{
    return std::max(0.0f, std::min(1.0f, x));
}

float clamp(float x, float minVal, float maxVal)
{
    return std::max(minVal, std::min(maxVal, x));
}

vec2 clamp(const vec2 &v, float minVal, float maxVal)
{
    return vec2(clamp(v.x, minVal, maxVal), clamp(v.y, minVal, maxVal));
}

vec3 clamp(const vec3 &v, float minVal, float maxVal)
{
    return vec3(clamp(v.x, minVal, maxVal), clamp(v.y, minVal, maxVal), clamp(v.z, minVal, maxVal));
}

float fract(float x)
{
    return x - std::floor(x);
}

vec2 fract(const vec2 &v)
{
    return vec2(fract(v.x), fract(v.y));
}

vec2 floor(const vec2 &v)
{
    return vec2(std::floor(v.x), std::floor(v.y));
}

vec3 floor(const vec3 &v)
{
    return vec3(std::floor(v.x), std::floor(v.y), std::floor(v.z));
}

vec3 abs(const vec3 &v)
{
    return vec3(std::abs(v.x), std::abs(v.y), std::abs(v.z));
}

vec2 abs(const vec2 &v)
{
    return vec2(std::abs(v.x), std::abs(v.y));
}

float dot(const vec2 &a, const vec2 &b)
{
    return a.x * b.x + a.y * b.y;
}

float dot(const vec3 &a, const vec3 &b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

float length(const vec2 &v)
{
    return std::sqrt(v.x * v.x + v.y * v.y);
}

float length(const vec3 &v)
{
    return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

vec2 normalize(const vec2 &v)
{
    float len = length(v);
    return len > 0 ? v / len : vec2(0, 0);
}

vec3 normalize(const vec3 &v)
{
    float len = length(v);
    return len > 0 ? v / len : vec3(0, 0, 0);
}

vec3 cross(const vec3 &a, const vec3 &b)
{
    return vec3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}

vec3 mix(const vec3 &a, const vec3 &b, float t)
{
    return a * (1.0f - t) + b * t;
}

vec3 mix(const vec3 &a, const vec3 &b, const vec3 &t)
{
    return vec3(
        a.x * (1.0f - t.x) + b.x * t.x,
        a.y * (1.0f - t.y) + b.y * t.y,
        a.z * (1.0f - t.z) + b.z * t.z);
}

float mix(float a, float b, float t)
{
    return a * (1.0f - t) + b * t;
}

vec2 mix(const vec2 &a, const vec2 &b, float t)
{
    return a * (1.0f - t) + b * t;
}

float smoothstep(float edge0, float edge1, float x)
{
    float t = clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

float step(float edge, float x)
{
    return x < edge ? 0.0f : 1.0f;
}

vec3 max(const vec3 &a, float b)
{
    return vec3(std::max(a.x, b), std::max(a.y, b), std::max(a.z, b));
}

vec2 max(const vec2 &a, float b)
{
    return vec2(std::max(a.x, b), std::max(a.y, b));
}

vec3 reflect(const vec3 &I, const vec3 &N)
{
    return I - N * (2.0f * dot(N, I));
}

// ============================================================================
// NOISE FUNCTIONS
// ============================================================================

float hash2D(const vec2 &p)
{
    return fract(std::sin(dot(p, vec2(127.1f, 311.7f))) * 43758.5453123f);
}

float hash(float n)
{
    return fract(std::sin(n) * 43758.5453f);
}

float noise2D(const vec2 &p)
{
    vec2 i = floor(p);
    vec2 f = fract(p);

    vec2 u = vec2(
        f.x * f.x * f.x * (f.x * (f.x * 6.0f - 15.0f) + 10.0f),
        f.y * f.y * f.y * (f.y * (f.y * 6.0f - 15.0f) + 10.0f));

    float a = hash2D(i);
    float b = hash2D(i + vec2(1.0f, 0.0f));
    float c = hash2D(i + vec2(0.0f, 1.0f));
    float d = hash2D(i + vec2(1.0f, 1.0f));

    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

vec2 worley2D(const vec2 &p)
{
    vec2 id = floor(p);
    vec2 fd = fract(p);

    float minDist = 1.0f;
    float secondMinDist = 1.0f;

    for (int y = -1; y <= 1; y++)
    {
        for (int x = -1; x <= 1; x++)
        {
            vec2 neighbor = vec2((float)x, (float)y);
            vec2 point = vec2(hash2D(id + neighbor));

            float dist = length(neighbor + point - fd);

            if (dist < minDist)
            {
                secondMinDist = minDist;
                minDist = dist;
            }
            else if (dist < secondMinDist)
            {
                secondMinDist = dist;
            }
        }
    }

    return vec2(minDist, secondMinDist);
}

// ============================================================================
// GERSTNER WAVE SYSTEM
// ============================================================================

vec2 wavedx(const vec2 &position, const vec2 &direction, float frequency, float timeshift)
{
    float x = dot(direction, position) * frequency + timeshift;
    float wave = std::exp(std::sin(x) - 1.0f);
    float dx = wave * std::cos(x);
    return vec2(wave, -dx);
}

float getWaves(vec2 position, int iterations, float time)
{
    float wavePhaseShift = length(position) * 0.1f;
    float iter = 0.0f;
    float frequency = 1.0f;
    float timeMultiplier = 2.0f;
    float weight = 1.0f;
    float sumOfValues = 0.0f;
    float sumOfWeights = 0.0f;

    for (int i = 0; i < iterations; i++)
    {
        vec2 p = vec2(std::sin(iter), std::cos(iter));
        vec2 res = wavedx(position, p, frequency, time * timeMultiplier + wavePhaseShift);

        position += p * res.y * weight * DRAG_MULT;

        sumOfValues += res.x * weight;
        sumOfWeights += weight;

        weight = mix(weight, 0.0f, 0.2f);
        frequency *= 1.18f;
        timeMultiplier *= 1.07f;
        iter += 1232.399963f;
    }

    return sumOfValues / sumOfWeights;
}

float getWaveHeight(const vec2 &worldPos, float time)
{
    float distFromOrigin = length(worldPos);
    float lodFactor = 1.0f - smoothstep(15.0f, 50.0f, distFromOrigin);
    float height = getWaves(worldPos, 8, time);
    float amplitudeScale = 0.25f * (0.1f + lodFactor * 0.8f);
    height = height * amplitudeScale - 0.15f;
    return height;
}

vec3 calculateWaveNormal(const vec2 &pos, float time, float epsilon)
{
    float H = getWaves(pos, 12, time);
    vec2 ex = vec2(epsilon, 0);

    vec3 a = vec3(pos.x, H, pos.y);
    vec3 b = vec3(pos.x - epsilon, getWaves(pos - ex, 12, time), pos.y);
    vec3 c = vec3(pos.x, getWaves(vec2(pos.x, pos.y + epsilon), 12, time), pos.y + epsilon);

    return normalize(cross(a - b, a - c));
}

// ============================================================================
// TOON SHADING UTILITIES
// ============================================================================

float toonShading(float intensity, int bands)
{
    float bandSize = 1.0f / (float)bands;
    float band = std::floor(intensity / bandSize);
    return band * bandSize;
}

vec3 applyToonLighting(const vec3 &baseColor, const vec3 &normal, const vec3 &lightDir, const vec3 &viewDir)
{
    float NdotL = std::max(0.0f, dot(normal, lightDir));
    float toonDiffuse = toonShading(NdotL, 4);

    vec3 halfVec = normalize(lightDir + viewDir);
    float specular = std::pow(std::max(0.0f, dot(normal, halfVec)), 32.0f);
    float toonSpecular = toonShading(specular, 2);

    float rim = 1.0f - std::max(0.0f, dot(normal, viewDir));
    rim = std::pow(rim, 3.0f);
    rim = step(0.65f, rim);

    vec3 ambient = baseColor * 0.4f;
    vec3 diffuse = baseColor * (0.5f + toonDiffuse * 0.35f);
    vec3 spec = vec3(1.0f, 1.0f, 1.0f) * toonSpecular * 0.4f;
    vec3 rimColor = vec3(0.3f, 0.35f, 0.4f) * rim * 0.15f;

    return clamp(ambient + diffuse + spec + rimColor, 0.0f, 1.0f);
}

// ============================================================================
// ENHANCED FOAM AND SPRAY
// ============================================================================

float calculateWaveCrestFoam(const vec3 &worldPos, const vec3 &normal, float time)
{
    float steepness = 1.0f - normal.y;
    steepness = smoothstep(0.15f, 0.35f, steepness);

    float foamNoise1 = noise2D(vec2(worldPos.x, worldPos.z) * 6.0f + vec2(time, time) * vec2(0.3f, 0.2f));
    float foamNoise2 = noise2D(vec2(worldPos.x, worldPos.z) * 12.0f - vec2(time, time) * vec2(0.2f, 0.3f));

    float foam = foamNoise1 * foamNoise2;
    foam = step(0.45f, foam);

    return foam * steepness;
}

float calculateSeaSpray(const vec3 &worldPos, const vec3 &normal, float time)
{
    float steepness = 1.0f - normal.y;
    if (steepness < 0.2f)
        return 0.0f;

    float spray = noise2D(vec2(worldPos.x, worldPos.z) * 8.0f + vec2(time, time) * vec2(0.8f, 0.6f));
    spray += noise2D(vec2(worldPos.x, worldPos.z) * 16.0f - vec2(time, time) * vec2(0.5f, 0.7f)) * 0.5f;
    spray = step(0.75f, spray);
    spray *= smoothstep(0.25f, 0.4f, steepness);

    return spray;
}

float calculateShoreFoam(float depth, float time, const vec2 &worldPos)
{
    const float foamThreshold = 1.2f;
    float foamDiff = saturate(depth / foamThreshold);

    float foamNoise = noise2D(worldPos * 5.0f + vec2(time, time) * vec2(0.5f, 0.4f));
    foamNoise += noise2D(worldPos * 10.0f - vec2(time, time) * vec2(0.3f, 0.5f)) * 0.5f;

    float foamLines = std::sin((foamDiff - time * 0.8f) * 8.0f * PI);
    foamLines = saturate(foamLines) * (1.0f - foamDiff);

    float foam = step(0.5f - foamLines * 0.3f, foamNoise);
    return foam * (1.0f - saturate(depth / foamThreshold));
}

// ============================================================================
// 2D CLOUD SHADER (FIXED - No more diamond artifacts)
// ============================================================================

const float cloudscale = 1.1f;
const float cloudSpeed = 0.03f;
const float clouddark = 0.5f;
const float cloudlight = 0.3f;
const float cloudcover = 0.2f;
const float cloudalpha = 8.0f;
const float skytint = 0.5f;
const vec3 skycolour1 = vec3(0.2f, 0.4f, 0.6f);
const vec3 skycolour2 = vec3(0.4f, 0.7f, 1.0f);

struct mat2
{
    float m[2][2];
    mat2(float a, float b, float c, float d)
    {
        m[0][0] = a;
        m[0][1] = b;
        m[1][0] = c;
        m[1][1] = d;
    }
    vec2 operator*(const vec2 &v) const
    {
        return vec2(
            m[0][0] * v.x + m[0][1] * v.y,
            m[1][0] * v.x + m[1][1] * v.y);
    }
};

const mat2 cloudMatrix = mat2(1.6f, 1.2f, -1.2f, 1.6f);

vec2 hashCloud(const vec2 &p)
{
    vec2 p1 = vec2(dot(p, vec2(127.1f, 311.7f)), dot(p, vec2(269.5f, 183.3f)));
    vec2 sinP = vec2(std::sin(p1.x) * 43758.5453123f, std::sin(p1.y) * 43758.5453123f);
    vec2 fractP = fract(sinP);
    return fractP * 2.0f - vec2(1.0f, 1.0f);
}

float noiseCloud(const vec2 &p)
{
    const float K1 = 0.366025404f;
    const float K2 = 0.211324865f;
    vec2 i = floor(p + vec2((p.x + p.y) * K1, (p.x + p.y) * K1));
    vec2 a = p - i + vec2((i.x + i.y) * K2, (i.x + i.y) * K2);
    vec2 o = (a.x > a.y) ? vec2(1.0f, 0.0f) : vec2(0.0f, 1.0f);
    vec2 b = a - o + vec2(K2, K2);
    vec2 c = a - vec2(1.0f, 1.0f) + vec2(2.0f * K2, 2.0f * K2);

    vec3 h = vec3(
        0.5f - dot(a, a),
        0.5f - dot(b, b),
        0.5f - dot(c, c));
    h = max(h, 0.0f);

    vec3 n = vec3(
        h.x * h.x * h.x * h.x * dot(a, hashCloud(i + vec2(0.0f, 0.0f))),
        h.y * h.y * h.y * h.y * dot(b, hashCloud(i + o)),
        h.z * h.z * h.z * h.z * dot(c, hashCloud(i + vec2(1.0f, 1.0f))));

    return dot(n, vec3(70.0f, 70.0f, 70.0f));
}

float fbmCloud(vec2 n)
{
    float total = 0.0f, amplitude = 0.1f;
    for (int i = 0; i < 7; i++)
    {
        total += noiseCloud(n) * amplitude;
        n = cloudMatrix * n;
        amplitude *= 0.4f;
    }
    return total;
}

vec3 renderClouds(const vec2 &fragCoord, float iResX, float iResY, float iTime)
{
    vec2 p = vec2(fragCoord.x / iResX, fragCoord.y / iResY);
    vec2 uv = vec2(p.x * (iResX / iResY), p.y);
    float time = iTime * cloudSpeed;
    float q = fbmCloud(uv * cloudscale * 0.5f);

    // Layer 1: Ridged noise
    float r = 0.0f;
    uv = vec2(p.x * (iResX / iResY), p.y) * cloudscale;
    uv = uv - vec2(q, q) - vec2(time, time);
    float weight = 0.8f;
    for (int i = 0; i < 8; i++)
    {
        r += std::abs(weight * noiseCloud(uv));
        uv = cloudMatrix * uv + vec2(time, time);
        weight *= 0.7f;
    }

    // Layer 2: Soft variation
    float f = 0.0f;
    uv = vec2(p.x * (iResX / iResY), p.y) * cloudscale;
    uv = uv - vec2(q, q) - vec2(time, time);
    weight = 0.7f;
    for (int i = 0; i < 8; i++)
    {
        f += weight * noiseCloud(uv);
        uv = cloudMatrix * uv + vec2(time, time);
        weight *= 0.6f;
    }

    f *= r + f;

    // Layer 3: Color modulation
    float c = 0.0f;
    time = iTime * cloudSpeed * 2.0f;
    uv = vec2(p.x * (iResX / iResY), p.y) * cloudscale * 2.0f;
    uv = uv - vec2(q, q) - vec2(time, time);
    weight = 0.4f;
    for (int i = 0; i < 7; i++)
    {
        c += weight * noiseCloud(uv);
        uv = cloudMatrix * uv + vec2(time, time);
        weight *= 0.6f;
    }

    // Layer 4: Ridged color detail
    float c1 = 0.0f;
    time = iTime * cloudSpeed * 3.0f;
    uv = vec2(p.x * (iResX / iResY), p.y) * cloudscale * 3.0f;
    uv = uv - vec2(q, q) - vec2(time, time);
    weight = 0.4f;
    for (int i = 0; i < 7; i++)
    {
        c1 += std::abs(weight * noiseCloud(uv));
        uv = cloudMatrix * uv + vec2(time, time);
        weight *= 0.6f;
    }

    c += c1;

    // Sky gradient
    vec3 skycolour = mix(skycolour2, skycolour1, p.y);

    // Improved toon quantization (smoother transitions)
    float cloudLuminance = clamp((clouddark + cloudlight * c), 0.0f, 1.0f);

    // Smooth stepped quantization instead of hard if-else
    float quantized = 0.25f;
    if (cloudLuminance > 0.25f)
        quantized = 0.45f;
    if (cloudLuminance > 0.35f)
        quantized = 0.65f;
    if (cloudLuminance > 0.65f)
        quantized = 0.8f;

    // Blend slightly between levels to reduce banding
    cloudLuminance = mix(cloudLuminance, quantized, 0.7f);

    vec3 cloudcolour = vec3(1.1f, 1.1f, 0.9f) * cloudLuminance;

    f = cloudcover + cloudalpha * f * r;

    // Smoother alpha transition
    float cloudAlpha = clamp(f + c, 0.0f, 1.0f);
    cloudAlpha = smoothstep(0.25f, 0.75f, cloudAlpha);

    // Final blend
    vec3 result = mix(skycolour, clamp(skycolour * skytint + cloudcolour, 0.0f, 1.0f), cloudAlpha);

    return result;
}

// ============================================================================
// BOAT WITH WIND-BLOWN FLAG
// ============================================================================

float sdBox(const vec3 &p, const vec3 &b)
{
    vec3 q = abs(p) - b;
    return length(max(q, 0.0f)) + std::min(std::max(q.x, std::max(q.y, q.z)), 0.0f);
}

float sdCylinder(const vec3 &p, float r, float h)
{
    vec2 d = abs(vec2(length(vec2(p.x, p.z)), p.y)) - vec2(r, h);
    return std::min(std::max(d.x, d.y), 0.0f) + length(max(d, 0.0f));
}

float sdRoundBox(const vec3 &p, const vec3 &b, float r)
{
    vec3 q = abs(p) - b;
    return length(max(q, 0.0f)) + std::min(std::max(q.x, std::max(q.y, q.z)), 0.0f) - r;
}

float smin(float a, float b, float k)
{
    float h = clamp(0.5f + 0.5f * (b - a) / k, 0.0f, 1.0f);
    return mix(b, a, h) - k * h * (1.0f - h);
}

float sdFlag(const vec3 &p, float time)
{
    vec3 flagP = p;
    flagP.y -= 0.75f;
    flagP.z -= 0.15f;
    flagP.x += 0.08f;

    float distFromPole = std::max(0.0f, flagP.x - 0.02f);
    float distFactor = (flagP.x - 0.02f) * 10.0f;

    float wave1 = std::sin(flagP.x * 12.0f + time * 4.5f) * 0.025f * distFactor;
    float wave2 = std::sin(flagP.x * 20.0f - time * 6.0f) * 0.015f * distFactor;
    float wave3 = std::sin(flagP.x * 35.0f + time * 8.0f) * 0.008f * distFactor;

    float verticalWave = std::sin(flagP.x * 10.0f + time * 5.0f) * 0.02f * distFactor;

    flagP.y += wave1 + wave2 + verticalWave;
    flagP.z += (wave1 + wave3) * 0.6f;

    float flag = sdBox(flagP, vec3(0.12f, 0.08f, 0.002f));
    return flag;
}

float sdBoat(const vec3 &p, float time)
{
    vec3 hullP = p;
    hullP.y += 0.08f;
    float hull = sdRoundBox(hullP, vec3(0.9f, 0.1f, 0.3f), 0.05f);

    vec3 bowP = p;
    bowP.z += 0.35f;
    bowP.y += 0.08f;
    float bow = sdBox(bowP, vec3(0.5f, 0.08f, 0.05f));
    hull = smin(hull, bow, 0.1f);

    vec3 sternP = p;
    sternP.z -= 0.35f;
    sternP.y += 0.08f;
    float stern = sdRoundBox(sternP, vec3(0.7f, 0.08f, 0.05f), 0.03f);
    hull = smin(hull, stern, 0.08f);

    vec3 deckP = p;
    deckP.y -= 0.02f;
    float deck = sdBox(deckP, vec3(0.85f, 0.02f, 0.28f));
    hull = smin(hull, deck, 0.04f);

    vec3 cabinP = p;
    cabinP.y -= 0.2f;
    cabinP.z -= 0.15f;
    float cabin = sdRoundBox(cabinP, vec3(0.4f, 0.15f, 0.3f), 0.03f);

    vec3 roofP = p;
    roofP.y -= 0.38f;
    roofP.z -= 0.15f;
    float roof = sdRoundBox(roofP, vec3(0.42f, 0.03f, 0.32f), 0.02f);
    cabin = smin(cabin, roof, 0.05f);

    vec3 windowP = p;
    windowP.y -= 0.22f;
    windowP.z -= 0.15f;
    windowP.x -= 0.25f;
    float window1 = sdBox(windowP, vec3(0.08f, 0.08f, 0.32f));

    windowP.x += 0.5f;
    float window2 = sdBox(windowP, vec3(0.08f, 0.08f, 0.32f));

    cabin = std::max(cabin, -window1 * 0.5f);
    cabin = std::max(cabin, -window2 * 0.5f);

    vec3 mastP = p;
    mastP.y -= 0.5f;
    mastP.z -= 0.2f;
    float mast = sdCylinder(mastP, 0.03f, 0.5f);

    vec3 nestP = p;
    nestP.y -= 0.85f;
    nestP.z -= 0.2f;
    float nest = sdCylinder(nestP, 0.08f, 0.04f);
    mast = smin(mast, nest, 0.03f);

    float flag = sdFlag(p, time);

    vec3 stackP = p;
    stackP.y -= 0.45f;
    stackP.z += 0.1f;
    stackP.x -= 0.2f;
    float stack = sdCylinder(stackP, 0.05f, 0.08f);

    float boat = smin(hull, cabin, 0.06f);
    boat = std::min(boat, mast);
    boat = std::min(boat, flag);
    boat = std::min(boat, stack);

    return boat;
}

int getBoatMaterialID(const vec3 &p, float time)
{
    vec3 hullP = p;
    hullP.y += 0.08f;
    float hull = sdRoundBox(hullP, vec3(0.9f, 0.1f, 0.3f), 0.05f);

    vec3 cabinP = p;
    cabinP.y -= 0.2f;
    cabinP.z -= 0.15f;
    float cabin = sdRoundBox(cabinP, vec3(0.4f, 0.15f, 0.3f), 0.03f);

    vec3 mastP = p;
    mastP.y -= 0.5f;
    mastP.z -= 0.2f;
    float mast = sdCylinder(mastP, 0.03f, 0.5f);

    float flag = sdFlag(p, time);

    if (flag < hull && flag < cabin && flag < mast)
        return 3;
    if (mast < hull && mast < cabin)
        return 2;
    if (cabin < hull)
        return 1;
    return 0;
}

vec3 estimateNormalBoat(const vec3 &p, float time)
{
    float eps = 0.001f;
    float dx = sdBoat(p + vec3(eps, 0, 0), time) - sdBoat(p - vec3(eps, 0, 0), time);
    float dy = sdBoat(p + vec3(0, eps, 0), time) - sdBoat(p - vec3(0, eps, 0), time);
    float dz = sdBoat(p + vec3(0, 0, eps), time) - sdBoat(p - vec3(0, 0, eps), time);
    return normalize(vec3(dx, dy, dz));
}

float intersectBoatSDF(const vec3 &ro, const vec3 &rd, const vec3 &center, float time)
{
    float t = 0.0f;
    const int MAX_STEPS = 100;
    const float EPS = 0.001f;
    const float MAX_DIST = 80.0f;
    for (int i = 0; i < MAX_STEPS; i++)
    {
        vec3 p = ro + rd * t;
        float d = sdBoat(p - center, time);
        if (d < EPS)
            return t;
        t += d;
        if (t > MAX_DIST)
            break;
    }
    return -1.0f;
}

vec3 shadeBoat(const vec3 &pos, const vec3 &normal, const vec3 &viewDir, const vec3 &boatCenter, float time)
{
    vec3 lightDir = normalize(vec3(0.7f, 0.6f, -0.3f));

    int matID = getBoatMaterialID(pos - boatCenter, time);

    vec3 base;
    if (matID == 0)
    {
        base = vec3(0.85f, 0.25f, 0.18f);
    }
    else if (matID == 1)
    {
        base = vec3(0.98f, 0.95f, 0.88f);
    }
    else if (matID == 2)
    {
        base = vec3(0.45f, 0.32f, 0.22f);
    }
    else
    {
        base = vec3(0.95f, 0.12f, 0.12f);
    }

    return applyToonLighting(base, normal, lightDir, viewDir);
}

// ============================================================================
// IMPROVED WATER RENDERING
// ============================================================================

float getBoatShadow(const vec3 &worldPos, const vec3 &boatCenter, const vec3 &lightDir, float time)
{
    vec3 shadowRayOrigin = worldPos + vec3(0.0f, 0.01f, 0.0f);
    vec3 shadowRayDir = lightDir;

    float t = 0.0f;
    const int steps = 20;
    for (int i = 0; i < steps; i++)
    {
        vec3 p = shadowRayOrigin + shadowRayDir * t;
        float d = sdBoat(p - boatCenter, time);
        if (d < 0.05f)
            return 0.3f;
        t += std::max(0.05f, d);
        if (t > 3.0f)
            break;
    }
    return 1.0f;
}

vec3 getBoatReflectionColor(const vec3 &worldPos, const vec3 &boatCenter, const vec3 &viewDir, const vec3 &normal, float time)
{
    vec3 reflectDir = reflect(viewDir, normal);

    vec3 mirroredBoatCenter = boatCenter;
    mirroredBoatCenter.y = -mirroredBoatCenter.y;

    float t = 0.0f;
    const int steps = 35;
    for (int i = 0; i < steps; i++)
    {
        vec3 p = worldPos + reflectDir * t;

        float d = sdBoat(p - mirroredBoatCenter, time);
        if (d < 0.08f)
        {
            int matID = getBoatMaterialID(p - mirroredBoatCenter, time);

            vec3 refNormal = estimateNormalBoat(p - mirroredBoatCenter, time);
            vec3 lightDir = normalize(vec3(0.7f, 0.6f, -0.3f));
            float refLight = std::max(0.3f, dot(refNormal, lightDir));

            if (matID == 0)
                return vec3(0.75f, 0.22f, 0.16f) * refLight;
            if (matID == 1)
                return vec3(0.9f, 0.88f, 0.82f) * refLight;
            if (matID == 2)
                return vec3(0.4f, 0.28f, 0.2f) * refLight;
            return vec3(0.92f, 0.15f, 0.12f) * refLight;
        }
        t += std::max(0.04f, d);
        if (t > 8.0f)
            break;
    }
    return vec3(0.0f, 0.0f, 0.0f);
}

vec3 renderWater(const vec3 &ro, const vec3 &rd, const vec3 &skyColor, const vec3 &boatCenter, float time)
{
    float t = 0.1f;
    const int maxSteps = 180;
    bool hit = false;
    vec3 hitPos;

    for (int i = 0; i < maxSteps; i++)
    {
        hitPos = ro + rd * t;
        float waveHeight = getWaveHeight(vec2(hitPos.x, hitPos.z), time);
        float diff = hitPos.y - waveHeight;

        float adaptiveEpsilon = 0.005f + t * 0.0002f;

        if (std::abs(diff) < adaptiveEpsilon)
        {
            hit = true;
            break;
        }

        float stepSize = std::max(0.008f, std::abs(diff) * 0.25f);
        t += stepSize;

        if (t > 120.0f)
            break;
    }

    if (!hit)
    {
        return skyColor;
    }

    float distance = t;

    float normalEps = 0.015f + distance * 0.0001f;
    vec3 normal = calculateWaveNormal(vec2(hitPos.x, hitPos.z), time, normalEps);

    float depth = std::min(t * 0.2f, 8.0f);

    // Micro perturbation
    vec2 microUV = vec2(hitPos.x, hitPos.z) * 8.0f + vec2(time, time) * 0.6f;

    float n1 = noise2D(microUV);
    float n2 = noise2D(microUV * 2.7f + 13.1f);
    float n3 = noise2D(microUV * 5.1f - 7.3f);

    vec2 m = vec2(n1, n2) * 2.0f - vec2(1.0f, 1.0f);
    m = m + vec2(n3, n3) * (2.0f - 1.0f) * 0.5f;

    float microStrength = 0.1f;

    vec3 microNormal = normalize(vec3(
        m.x * microStrength,
        1.0f,
        m.y * microStrength));

    float blend = 0.15f;
    normal = normalize(mix(normal, microNormal, blend));

    vec3 shallowColor = vec3(0.5f, 0.75f, 0.95f);
    vec3 midColor = vec3(0.3f, 0.65f, 0.85f);
    vec3 deepColor = vec3(0.15f, 0.45f, 0.7f);
    vec3 veryDeepColor = vec3(0.08f, 0.25f, 0.5f);

    vec2 worley = worley2D(vec2(hitPos.x, hitPos.z) * 2.0f + vec2(time, time) * 0.15f);
    float caustics = worley.y - worley.x;
    caustics = smoothstep(0.1f, 0.4f, caustics) * 0.6f;

    vec2 worleyLarge = worley2D(vec2(hitPos.x, hitPos.z) * 0.2f + vec2(time, time) * 0.03f);
    float deepPattern = smoothstep(0.3f, 0.7f, worleyLarge.x);

    vec3 baseColor;
    float depthT = smoothstep(0.0f, 6.0f, depth);

    baseColor = mix(shallowColor, midColor, smoothstep(0.0f, 0.3f, depthT));
    baseColor = mix(baseColor, deepColor, smoothstep(0.3f, 0.7f, depthT));
    baseColor = mix(baseColor, veryDeepColor, smoothstep(0.7f, 1.0f, depthT));

    float causticsStrength = 1.0f - smoothstep(0.0f, 1.5f, depth);

    caustics = std::pow(caustics, 0.8f);
    baseColor += vec3(0.25f, 0.3f, 0.35f) * caustics * causticsStrength;

    vec3 lightDir = normalize(vec3(0.7f, 0.6f, -0.3f));
    vec3 viewDir = rd * -1.0f;
    vec3 shadedColor = applyToonLighting(baseColor, normal, lightDir, viewDir);

    float shadow = getBoatShadow(hitPos, boatCenter, lightDir, time);
    if (shadow < 1.0f)
    {
        shadedColor = shadedColor * (0.6f + shadow * 0.4f);
    }

    vec3 reflectionColor = getBoatReflectionColor(hitPos, boatCenter, rd, normal, time);
    if (length(reflectionColor) > 0.0f)
    {
        float reflectionStrength = 0.65f;
        float distortion = noise2D(vec2(hitPos.x, hitPos.z) * 3.0f + vec2(time, time) * 0.5f) * 0.1f;
        reflectionStrength *= (1.0f - distortion);

        shadedColor = mix(shadedColor, reflectionColor, reflectionStrength);

        float ripple = smoothstep(0.5f, 0.7f, noise2D(vec2(hitPos.x, hitPos.z) * 8.0f + time));
        shadedColor += reflectionColor * ripple * 0.2f;
    }

    float fresnel = std::pow(1.0f - std::max(0.0f, dot(normal, viewDir)), 3.0f);
    fresnel = toonShading(fresnel, 2);
    shadedColor = mix(shadedColor, skyColor * 0.8f, fresnel * 0.15f);

    float shoreFoam = calculateShoreFoam(depth, time, vec2(hitPos.x, hitPos.z));
    float crestFoam = calculateWaveCrestFoam(hitPos, normal, time);
    float spray = calculateSeaSpray(hitPos, normal, time);

    float totalFoam = std::max(shoreFoam, std::max(crestFoam, spray));

    vec3 foamColor = vec3(0.95f, 0.98f, 1.0f);
    shadedColor = mix(shadedColor, foamColor, totalFoam * 0.85f);
    shadedColor += vec3(1.0f, 1.0f, 1.0f) * spray * 0.3f;

    float distanceFade = smoothstep(70.0f, 115.0f, distance);
    shadedColor = mix(shadedColor, skyColor, distanceFade * 0.5f);

    return shadedColor;
}

float intersectWater(const vec3 &ro, const vec3 &rd, vec3 &outHitPos, float iTime)
{
    float t = 0.1f;
    const int maxSteps = 100;
    const float epsilon = 0.05f;
    for (int i = 0; i < maxSteps; i++)
    {
        vec3 pos = ro + rd * t;
        float waveHeight = getWaveHeight(vec2(pos.x, pos.z), iTime);
        float diff = pos.y - waveHeight;
        if (diff < epsilon && diff > -epsilon)
        {
            outHitPos = pos;
            return t;
        }
        t += std::max(0.02f, std::abs(diff) * 0.5f);
        if (t > 80.0f)
            break;
    }
    return -1.0f;
}

// ============================================================================
// MAIN IMAGE
// ============================================================================

void mainImage(vec4 &fragColor, const vec2 &fragCoord, float iResX, float iResY, float iTime)
{
    vec2 uv = (fragCoord * 2.0f - vec2(iResX, iResY)) / iResY;

    vec3 ro = vec3(0.0f, 1.5f, -4.0f);
    vec3 rd = normalize(vec3(uv.x, uv.y, 1.0f));

    vec3 cloudColor = renderClouds(fragCoord, iResX, iResY, iTime);
    vec3 finalColor = cloudColor;

    vec2 objXZ = vec2(0.3f, -1.5f);
    float driftSpeed = 0.15f;
    float driftAmt = 0.5f;

    objXZ += vec2(std::sin(iTime * driftSpeed) * driftAmt,
                  std::cos(iTime * driftSpeed * 0.7f) * (driftAmt * 0.6f));

    float objWaveY = getWaveHeight(objXZ, iTime);
    float bob = noise2D(objXZ * 1.5f + vec2(iTime, iTime) * 0.8f) * 0.05f;

    vec3 boatCenter = vec3(objXZ.x, objWaveY + 0.15f + bob, objXZ.y);

    vec3 waterHitPos;
    float tWater = intersectWater(ro, rd, waterHitPos, iTime);

    float tBoat = intersectBoatSDF(ro, rd, boatCenter, iTime);

    if (tBoat > 0.0f && (tWater < 0.0f || tBoat < tWater))
    {
        vec3 surfPos = ro + rd * tBoat;
        vec3 surfNormal = estimateNormalBoat(surfPos - boatCenter, iTime);
        vec3 viewDir = rd * -1.0f;
        vec3 boatColor = shadeBoat(surfPos, surfNormal, viewDir, boatCenter, iTime);

        finalColor = boatColor;
    }
    else
    {
        if (rd.y < 0.5f)
        {
            vec3 waterColor = renderWater(ro, rd, cloudColor, boatCenter, iTime);
            if (dot(waterColor, waterColor) > 0.0f)
            {
                finalColor = waterColor;
            }
        }
    }

    fragColor = vec4(finalColor.x, finalColor.y, finalColor.z, 1.0f);
}

// ============================================================================
// RENDERING AND FILE I/O
// ============================================================================

void savePPM(const char *filename, int width, int height, const std::vector<unsigned char> &pixels)
{
    std::ofstream file(filename, std::ios::binary);
    file << "P6\n"
         << width << " " << height << "\n255\n";
    file.write(reinterpret_cast<const char *>(pixels.data()), pixels.size());
    file.close();
    std::cout << "  Saved: " << filename << std::endl;
}

void render(int width, int height, float time, std::vector<unsigned char> &pixels)
{
    int pixelCount = 0;
    int totalPixels = width * height;
    int progressStep = totalPixels / 20;

    std::cout << "  Progress: [";

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            vec2 fragCoord = vec2((float)x, (float)(height - 1 - y));
            vec4 color;
            mainImage(color, fragCoord, (float)width, (float)height, time);

            int idx = (y * width + x) * 3;
            pixels[idx + 0] = (unsigned char)(clamp(color.x, 0.0f, 1.0f) * 255.0f);
            pixels[idx + 1] = (unsigned char)(clamp(color.y, 0.0f, 1.0f) * 255.0f);
            pixels[idx + 2] = (unsigned char)(clamp(color.z, 0.0f, 1.0f) * 255.0f);

            pixelCount++;
            if (pixelCount % progressStep == 0)
            {
                std::cout << "=" << std::flush;
            }
        }
    }

    std::cout << "] Done" << std::endl;
}

// ============================================================================
// MAIN - Program entry point
// ============================================================================

int main()
{
    // Render settings
    const int width = 800;
    const int height = 600;
    const int numFrames = 60;
    const float fps = 30.0f;
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
    for (int frame = 0; frame < numFrames; frame++)
    {
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
    std::cout << "To create video:" << std::endl;
    std::cout << "ffmpeg -framerate 30 -i output/ocean_frame_%04d.ppm -c:v libx264 -pix_fmt yuv420p -crf 18 ocean_animation.mp4" << std::endl;

    return 0;
}