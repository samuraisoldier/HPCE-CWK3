CPPFLAGS += -I include
CPPFLAGS += -W -Wall
CPPFLAGS += -std=c++11
CPPFLAGS += -O3

# LDLIBS += -lOpenCL

all : bin/make_world bin/render_world bin/step_world bin/hs2715/step_world_v1_lambda bin/hs2715/StepWorldV2Function bin/hs2715/StepWorldV3OpenCL bin/hs2715/StepWorldV4DoubleBuffered

bin/% : src/%.cpp src/heat.cpp
	mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS)

bin/test_opencl : src/test_opencl.cpp
	mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LDFLAGS) -lOpenCL
	
	
bin/hs2715/%: src/hs2715/%.cpp src/heat.cpp
	mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) -o $@ $^ $(LDFLAGS) -lOpenCL
