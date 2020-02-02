- [ ] Rename all the things!
- [ ] We will need multi-input processor nodes.
- [ ] Interpolated noise (animation parameters) generation into its own process
- [ ] Do ImageFX in cupy.
- [ ] Animate latent space exploration in BigGAN
  - [ ] Create a single latent vector.
  - [ ] When an animation is triggered, create a second latent vector,
  - [ ] interpolate between these two vectors, when animation reaches its final frame, stop.
  - [ ] Put final frame into the first frame.
  - [ ] Do the same for labels. But instead of random, start and end labels will be selected from TouchOSC.
- [ ] Create 2 OSCClients
      One will relay unhandled messages to Onur
      Other will send initialization values to TouchOSC