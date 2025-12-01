# Stylized Ocean Scene Renderer

**CSCI 580 Final Project - Toon-Shaded Ocean with Dynamic Clouds and Boat**

## Team Members

- **Baidi Wang** - baidiwan@usc.edu
- **Dizhe Xiang** - dizhexia@usc.edu
- **Rae Chen** - raechen@usc.edu
- **Zhiqi Chen** - chenzhiq@usc.edu

## Project Overview

This is a real-time stylized ocean scene combining cel-shaded/toon rendering with realistic wave dynamics and atmospheric effects. It features:

- **Toon-shaded water** with discrete color bands and Worley noise patterns
- **Gerstner wave simulation** for realistic multi-directional ocean motion
- **Procedural 2D clouds** with cel-shaded lighting bands
- **Detailed boat model** with wind-animated flag and water reflections
- **Underwater caustics** and translucent water effects

## Features Implemented

### Wave System

- **Gerstner Wave Equations**: Physically-based wave simulation with:
  - Multiple overlapping waves at different frequencies
  - Amplitude and wavelength control
  - Phase shifts for natural variation
  - Drag multiplier for realistic wave interaction
- **Dynamic Foam System**:
  - Shore foam near shallow areas
  - Wave crest foam on steep slopes
  - Sea spray particles on wave peaks
- **Normal Calculation**: Computed from wave height field for accurate lighting

### Water Rendering

- **Multi-depth Color Gradients**:
  - Shallow (turquoise) → Mid (blue) → Deep (dark blue) transitions
  - Smoothstep blending between depth zones
- **Worley Noise Patterns**:
  - Caustic effects in shallow water
  - Large-scale patterns for deep water variation
  - Animated for dynamic underwater appearance
- **Toon Lighting**: 4-band diffuse + 2-band specular cel-shading
- **Fresnel Effects**: Angle-dependent reflections quantized for toon style
- **Horizon Blending**: Smooth transition to distant ocean color

### Cloud System

- **2D Procedural Clouds**: Layered noise-based approach
  - Ridged noise for cloud structure
  - Smooth noise for soft edges
  - Multiple octaves for detail variation
- **Cel-Shaded Lighting**: 3-band quantization (bright/mid/shadow)
- **Sky Gradient**: Two-tone blue gradient from horizon to zenith
- **Animation**: Clouds drift using matrix-based noise advection
- **Adjustable Parameters**: Coverage, darkness, alpha, and speed controls

### Boat Model

- **SDF-Based Geometry**: Signed Distance Field modeling
  - Hull with rounded bow and stern
  - Cabin with windows and roof
  - Mast with crow's nest
  - Smokestack
- **Wind-Animated Flag**:
  - Multiple sine wave frequencies
  - Exponential amplitude increase from pole to edge
  - Vertical and horizontal flutter motion
- **Material System**: 4 materials (hull, cabin, mast, flag) with distinct colors
- **Shadow Casting**: Soft shadows on water surface
- **Water Reflection**: Ray-marched reflection with distortion

### Toon/Cel Shading

- **Discrete Color Bands**: Quantized lighting in 3-4 levels
- **Rim Lighting**: Edge emphasis with step function
- **Stylized Specular**: Toon-shaded highlights
- **Band Thresholds**: Adjustable for different cel-shading intensities

## Technical Implementation

### Rendering Pipeline

1. **Sky & Clouds**: 2D screen-space procedural rendering
2. **Water Surface**: Ray-march Gerstner wave height field
3. **Boat**: SDF ray-marching with material ID system
4. **Compositing**: Depth-based layer ordering

### Shader Architecture

- **GLSL Fragment Shader** for Shadertoy/WebGL
- **Procedural Generation**: No texture dependencies
- **Real-time Performance**: Optimized ray-marching and noise functions

## Compilation & Running

### Running on Shadertoy

1. Copy `toon_ocean_final.glsl` to [Shadertoy.com](https://www.shadertoy.com/)
2. Click play to see real-time animation
3. Adjust parameters in the shader code for customization

### Basic Compilation - C++

```bash
# Create output directory
mkdir -p output

# Compile
g++ -std=c++17 -O2 toon_wave_shader.cpp -o toon_wave_shader

# Run the program
./toon_wave_shader

# Make the video (brew install ffmpeg)
ffmpeg -framerate 30 -i output/ocean_frame_%04d.ppm -c:v libx264 output/ocean_animation.mp4
```

## Credits & References

### Wave & Water Rendering

- **Gerstner Waves**: GPU Gems chapter on water simulation
  - Physically-based wave equations for realistic ocean motion
  - [GPU Gems - Chapter 1: Effective Water Simulation](https://developer.nvidia.com/gpugems/gpugems/part-i-natural-effects/chapter-1-effective-water-simulation-physical-models)
- **Fresnel Effect**: Schlick's approximation for view-angle dependent reflections
  - Used for water surface reflectivity based on viewing angle
  - [Schlick's Approximation](https://en.wikipedia.org/wiki/Schlick%27s_approximation)

### Procedural Noise & Clouds

- **Cloud Noise**: Shadertoy cloud rendering techniques
  - 2D simplex noise implementation
  - Fractional Brownian Motion (fBm) for cloud detail
  - Inspired by ["Clouds" by iq on Shadertoy](https://www.shadertoy.com/view/XslGRr)
- **Worley Noise**: Cellular noise for caustics and water patterns
  - [Worley Noise (Cellular Noise)](https://en.wikipedia.org/wiki/Worley_noise)
  - Used for underwater caustic light patterns

### SDF & Ray Marching

- **SDF Modeling**: Íñigo Quílez (iq) distance function library
  - [2D/3D Distance Functions](https://iquilezles.org/articles/distfunctions/)
  - Used for boat geometry (hull, cabin, mast, flag)
- **Ray Marching**: Sphere tracing technique for SDF rendering
  - [Ray Marching and Signed Distance Functions](https://iquilezles.org/articles/raymarchingdf/)

### Toon/Cel Shading

- **NPR (Non-Photorealistic Rendering)**: Discrete lighting bands
  - Quantization of diffuse and specular lighting
  - Step functions for hard shadow transitions
  - Inspired by games like _The Legend of Zelda: Wind Waker_ and _Genshin Impact_
- **Rim Lighting**: Edge detection for toon outlines
  - View-dependent edge highlighting

### Mathematical Functions

- **Perlin Noise**: Quintic interpolation for smooth value noise
  - Used for wave detail and foam patterns
- **Smoothstep**: Hermite interpolation for smooth transitions
  - Used throughout for color gradients and blending

### Tools & Libraries

- **Shadertoy**: Real-time GLSL shader development platform
  - [Shadertoy.com](https://www.shadertoy.com/)
- **FFmpeg**: Video encoding from image sequences
  - [FFmpeg.org](https://ffmpeg.org/)

## License

Educational project for CSCI 580 - Computer Graphics
