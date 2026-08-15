# Construct Spectrometer - Existing Design
design = Spectrometer(FULL_ENERGIES, 
                      measure_type="counts",
                      clip_range=(0, 100),
                      build_engine="NIST",
                      seperator=Layer(mylar, 0.5))

# Add layers
design.add_layer(Layer(quartz, 12.0))

for i in range(5):
    design.add_layer(Layer(lyso, 2.0))

for i in range(5):
    design.add_layer(Layer(w, 2.0))
    design.add_layer(Layer(lyso, 2.0))

# Finalise Design - Response Matrix, Namings
design.finalise()
