# Stylized Ocean Scene Renderer
**CSCI 580 Final Project - Toon-Shaded Ocean with Volumetric Clouds**

## Project Overview
This is a stylized ocean scene combining cel-shaded/toon rendering with animated wave dynamics. It features:
- **Toon-shaded water** with discrete color bands
- **Procedurally generated Gerstner waves** for realistic multi-directional ocean motion
- **Volumetric clouds** using ray-marching and procedural noise
- **Stylized reflections and foam** on wave surfaces
- Wind Waker / anime-inspired aesthetic

## Features Implemented

### Wave System
- **Gerstner Wave Equations**: Multiple overlapping waves with different:
  - Amplitudes (wave heights)
  - Wavelengths (distance between peaks)
  - Speeds and directions
  - Steepness factors
- **Dynamic Foam**: Appears on steep slopes and wave crests
- **Normal Calculation**: For accurate lighting using tangent/binormal vectors

### Cloud System
- **Volumetric Ray Marching**: Accurate 3D cloud rendering
- **Procedural Noise**: 
  - Worley noise for base cloud shapes
  - Fractal Brownian Motion (FBM) for detail
  - Perlin noise for smooth transitions
- **Light Scattering**: Beer's Law for realistic light attenuation
- **Animation**: Clouds drift across the sky with individual velocities

### Toon/Cel Shading
- **Discrete Color Bands**: Quantized lighting levels
- **Rim Lighting**: Edge emphasis for cartoon aesthetic
- **Stylized Specular**: Sharp highlights
- **Fresnel Effects**: Angle-dependent reflections

## Compilation & Running

### Basic Compilation
```bash
# Compile with C++11 support
g++ -std=c++11 ocean_scene.cpp -o ocean_scene -O3

# Run the program
./ocean_scene
```

### Optimized Compilation (Faster Rendering)
```bash
# Use maximum optimization and native CPU instructions
g++ -std=c++11 ocean_scene.cpp -o ocean_scene -O3 -march=native -ffast-math

# Run
./ocean_scene
```

### Expected Output
The program will:
1. Display rendering progress for each frame
2. Create PPM image files: `ocean_frame_0000.ppm`, `ocean_frame_0001.ppm`, etc.
3. Show completion message with video creation instructions

## Customization

### Render Settings (in `main()` function)
```cpp
const int width = 1920;          // Image width (try 800, 1920, 3840)
const int height = 1080;         // Image height (try 600, 1080, 2160)
const int numFrames = 60;        // Number of frames (1 for single image)
const float fps = 30.0f;         // Frames per second
```

### Wave Parameters (in `createOceanWaves()`)
Adjust individual wave properties:
```cpp
waves.push_back({
    0.15f,                              // amplitude (height)
    2.0f,                               // wavelength (size)
    0.8f,                               // speed
    Vec3(1.0f, 0.0f, 0.3f).normalize(), // direction
    0.4f                                // steepness (0-1)
});
```

### Toon Shading Bands (in `renderOcean()`)
```cpp
// Apply toon shading (3 diffuse bands, 2 specular bands)
Vec3 shadedColor = applyToonShading(waterColor, hit.normal, lightDir, viewDir, 
                                    3,  // diffuseBands (try 2-5)
                                    2); // specularBands (try 1-3)
```

### Cloud Density (in `createCloudClusters()`)
```cpp
const int numClusters = 18;  // Number of cloud clusters (try 10-30)
```

### Camera Position (in `render()`)
```cpp
Vec3 ro(0, 0.8f, -3.0f);  // Camera position (x, y, z)
                          // Adjust y for height, z for distance
```

## Converting Output Files

### Single Frame to PNG/JPG
```bash
# Convert to PNG (requires ImageMagick)
convert ocean_frame_0000.ppm ocean_frame_0000.png

# Convert to JPG
convert ocean_frame_0000.ppm ocean_frame_0000.jpg
```

### Create Animation Video
```bash
# Create MP4 video from frames (requires ffmpeg)
ffmpeg -framerate 30 -i ocean_frame_%04d.ppm -c:v libx264 -pix_fmt yuv420p ocean_animation.mp4

# Higher quality
ffmpeg -framerate 30 -i ocean_frame_%04d.ppm -c:v libx264 -crf 18 -pix_fmt yuv420p ocean_hq.mp4

# Create GIF (for web)
ffmpeg -framerate 30 -i ocean_frame_%04d.ppm -vf "scale=800:-1" ocean.gif
```

### Batch Convert All Frames
```bash
# Convert all PPM files to PNG
for f in ocean_frame_*.ppm; do convert "$f" "${f%.ppm}.png"; done
```

## Performance Tips

### For Faster Previews
- Reduce resolution: `width = 800, height = 600`
- Reduce ray marching steps:
  - In `raymarchClouds()`: Change `maxSteps = 250` to `100`
  - In `rayMarchOcean()`: Change `maxSteps = 100` to `50`
- Render single frame: `numFrames = 1`

### For Production Quality
- Increase resolution: `width = 1920, height = 1080` or `3840 x 2160`
- Keep default ray marching steps
- Enable compiler optimizations: `-O3 -march=native -ffast-math`

## Technical Details

### Coordinate System
- **X-axis**: Horizontal (left/right)
- **Y-axis**: Vertical (up/down)
- **Z-axis**: Depth (forward/back)

### Noise Functions
- **Perlin Noise**: Smooth gradient noise for natural variations
- **Worley Noise**: Cellular patterns for cloud base shapes
- **FBM**: Layered noise for fractal detail

### Rendering Pipeline
1. **Ray Generation**: Create rays from camera through each pixel
2. **Ocean Intersection**: Ray-march to find water surface
3. **Wave Displacement**: Apply Gerstner wave equation
4. **Ocean Shading**: Apply toon shading with foam and reflections
5. **Cloud Rendering**: Volumetric ray-march through cloud layer
6. **Sky Gradient**: Blend background sky color
7. **Compositing**: Combine all elements into final pixel color

## Project Structure

```
ocean_scene.cpp
├── Vector3 Class           - 3D vector mathematics
├── Utility Functions       - Math helpers (smoothstep, mix, etc.)
├── Noise Functions         - Procedural noise generation
│   ├── hash3()            - 3D random vectors
│   ├── noise3D()          - Perlin noise
│   ├── fbm()              - Fractal Brownian Motion
│   └── worley3D()         - Cellular/Voronoi noise
├── Cloud System
│   ├── CloudCluster       - Cloud data structure
│   ├── createCloudClusters() - Generate random clouds
│   ├── singleCloudDensity()  - Density for one cloud
│   ├── getCloudDensity()     - Combined cloud density
│   ├── lightMarch()          - Shadow calculations
│   └── raymarchClouds()      - Volumetric rendering
├── Wave System
│   ├── GerstnerWave       - Wave parameters
│   ├── createOceanWaves() - Define wave set
│   ├── gerstnerWave()     - Calculate displacement
│   ├── getOceanHeight()   - Height at position
│   └── calculateFoam()    - Foam generation
├── Toon Shading
│   ├── toonShading()      - Quantize lighting
│   └── applyToonShading() - Full toon shader
├── Ocean Rendering
│   ├── OceanHit           - Intersection data
│   ├── rayMarchOcean()    - Find surface
│   └── renderOcean()      - Shade water surface
└── Main Rendering
    ├── render()           - Main render loop
    ├── savePPM()          - File output
    └── main()             - Program entry
```

## Troubleshooting

### Compilation Errors
- **Error: "M_PI not defined"**: Add `#define _USE_MATH_DEFINES` before includes, or replace `M_PI` with `3.14159265358979323846`
- **C++11 features not recognized**: Ensure you're using `-std=c++11` flag

### Runtime Issues
- **Rendering too slow**: Reduce resolution or ray marching steps
- **Out of memory**: Reduce `numFrames` or render in batches
- **Strange colors**: Check that values are clamped between 0-255

### Image Quality
- **Jagged edges**: Increase resolution or implement anti-aliasing
- **Banding in gradients**: Increase toon shading bands
- **Clouds too sparse**: Increase `numClusters` or adjust density multiplier
- **Waves too calm**: Increase wave amplitude values

## Future Enhancements

Potential additions for extending this project:
- [ ] Add floating objects (boats, buoys) with physics
- [ ] Implement screen-space reflections
- [ ] Add underwater caustics
- [ ] Include sun/moon with god rays
- [ ] Add particle effects (spray, splashes)
- [ ] Implement outline rendering (edge detection)
- [ ] Add interactive camera controls
- [ ] Optimize with compute shaders

## Team Members
- Dizhe Xiang (dizhexia@usc.edu)
- Zhiqi Chen (chenzhiq@usc.edu)
- Baidi Wang (baidiwan@usc.edu)
- Rae Chen (raechen@usc.edu)

## References
- Gerstner Wave Equation: GPU Gems Chapter 1
- Toon Shading: Gooch et al. "A Non-Photorealistic Lighting Model For Automatic Technical Illustration"
- Procedural Noise: Perlin, K. "Improving Noise"
- Ocean Rendering: Shadertoy community examples
- Beer's Law: Physics-based volume rendering

---
*CSCI 580 - Computer Graphics Final Project*
