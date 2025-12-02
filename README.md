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

### Cloud System

We generate clouds procedurally using layered noise - combining ridged patterns for structure and smooth patterns for softness. The lighting is cel-shaded with discrete brightness bands to match our toon aesthetic. Clouds move over time using noise transformation, and blend naturally with the gradient sky background.

### Wave System

Our ocean uses Gerstner wave equations, which build complex motion by layering waves from different directions with varying frequencies. A drag coefficient creates the choppy, realistic peaks. We reduce wave detail at distance to maintain performance - near waves are full complexity, far waves are simplified. Surface normals come from the wave height field for lighting, and foam appears 
on steep crests and shallow areas.

### Boat Model

We build the boat using Signed Distance Functions - the hull, cabin, mast, and flag are all defined mathematically and rendered through ray marching. Each part has its own material color. The flag animates with sine waves that get stronger toward the free end, giving it a realistic flutter. The boat follows the wave motion beneath it and adds a gentle drift for natural movement.

### Boat Reflection

To create reflections, we flip the boat vertically below the water surface and ray march toward it using the reflected view direction. The reflection ray accounts for the water surface normal. We add noise-based distortion to make it look wavy and natural, then blend the reflection with the water color.

## Compilation & Running

### Running on Shadertoy

1. Copy `toon_wave_shader.glsl` to [Shadertoy.com](https://www.shadertoy.com/)
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
ffmpeg -framerate 30 -i output/ocean_frame_%04d.ppm -c:v libx264 -pix_fmt yuv420p -crf 18 output/ocean_animation.mp4

# Make the gif
ffmpeg -framerate 30 -i output/ocean_frame_%04d.ppm -vf "fps=30,scale=800:-1:flags=lanczos" output/ocean_animation.gif
```

## Credits & References

### Wave & Water Rendering

- **Gerstner Waves**:
  - Physically-based wave equations for realistic ocean motion
- **Fresnel Effect**: 
  - Used for water surface reflectivity based on viewing angle

### Procedural Noise & Clouds

- **Cloud Noise**: Shadertoy cloud rendering techniques
  - 2D simplex noise implementation
  - Fractional Brownian Motion (fBm) for cloud detail
  - Inspired by ["2D Clouds" by drift on Shadertoy](https://www.shadertoy.com/view/4tdSWr)
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
- **Rim Lighting**: Edge detection for toon outlines
  - View-dependent edge highlighting

### Other Functions

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
