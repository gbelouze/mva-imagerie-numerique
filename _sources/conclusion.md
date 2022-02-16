# Conclusion

To what extent is the guided filter fusion applicable to real world use case ?  

For HDR, most cameras are already capable to take at once several photographs with different exposition parameters. Granted we are able to tune hyperparameters adaptatively (which still is an open problem), the method could be used for fusion.  
For multi-focus fusion however, we saw how challenging it was to obtain good photographs. First, few cameras (none except lytro ?) are able to take at once several photographs with different focal lengths. Second, it requires a preprocessing registration step that is not obvious to design systematically.  

In both cases anyway, the computations are too slow to be used in real-time applications (for instance as a smartphone app), or at least our implementation is. For reference, it takes around 10 seconds to do the fusion of our 8 pen images.

**References**
```{bibliography}
```