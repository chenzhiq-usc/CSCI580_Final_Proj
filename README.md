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
g++ -std=c++11 ocean_scene_fixed.cpp -o ocean_scene_fixed -O3 -march=native

# Run the program
./ocean_scene_fixed

# Make the video (brew install ffmpeg)
ffmpeg -framerate 30 -i output/ocean_frame_%04d.ppm -c:v libx264 output/ocean.mp4
```
