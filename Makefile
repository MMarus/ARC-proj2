# /**
# * @File        Makefile
# * @Author      Jiri Jaros, Filip Vaverka, Vojtech Nikl
# * @Affiliation FIT, Brno University of Technology
# * @Email       jarosjir@fit.vutbr.cz
# * @Comments    Linux makefile for Salomon
# * 
# * @Tool        ARC project 2017
# * @Created     10 April  2015, 10:49 AM
# * @LastModif   27 March  2017, 11:17 PM
#
# */


# Load following modules first for icpc
# module load intel/2017.00 HDF5/1.8.16-intel-2017.00


#SET PATHS
#HDF5_DIR=/usr/local/hdf5-serial

#HDF5_DIR=/apps/all/HDF5/1.8.16-intel-2017.00
HDF5_DIR=/usr/

HDF5_MIC_DIR=../hdf5-mic/install

#SET COMPILATOR, FLAGS and LIBS
CXX = mpic++

CXXFLAGS=-W -Wall -Wextra -pedantic \
         -O3 -std=c++11 \
	 -DPARALLEL_IO \
	 -I. \
         -fopenmp
CXXFLAGS_NOMIC=-march=native -I$(HDF5_DIR)/include
CXXFLAGS_MIC=-mmic -I$(HDF5_MIC_DIR)/include

LDFLAGS=-O3 -std=c++11 -fopenmp
LDFLAGS_NOMIC=-march=native -L$(HDF5_DIR)/lib/ -Wl,-rpath,$(HDF5_DIR)/lib/
LDFLAGS_MIC=-mmic -L$(HDF5_MIC_DIR)/lib/ -Wl,-rpath,$(HDF5_MIC_DIR)/lib/

LIBS=-lhdf5

TARGET=arc_proj02
TARGET_MIC=arc_proj02_mic

all:	$(TARGET)

$(TARGET): proj02.o MaterialProperties.o BasicRoutines.o CPUMatrices.o
	$(CXX) $(LDFLAGS) $(LDFLAGS_NOMIC) $^ $(LIBS) -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(CXXFLAGS_NOMIC) -c -o $@ $<

$(TARGET_MIC): proj02_mic.o MaterialProperties_mic.o BasicRoutines_mic.o
	$(CXX) $(LDFLAGS) $(LDFLAGS_MIC) $^ $(LIBS) -o $@

%_mic.o: %.cpp
	$(CXX) $(CXXFLAGS) $(CXXFLAGS_MIC) -c -o $@ $<

test: $(TARGET)
	./arc_proj02 -n 1 -n 100 -t 4 -m 0 -w 100 -i ../DataGenerator/material.h5 -o result.h5

test1: $(TARGET)
	mpirun -np 16 ./arc_proj02 -n 100 -m 1 -w 100 -i ../DataGenerator/material.h5 -o result.h5 -v

test2: $(TARGET)
	mpirun -np 1 ./arc_proj02 -n 1000 -t 16 -m 1 -w 100 -i ../DataGenerator/material.h5 -o result.h5 -v

testmic:
	./arc_proj02_mic -n 100 -m 0 -w 10 -i ../DataGenerator/material.h5 -o result.h5
clean:
	rm -f *.o
	rm -f *~
	rm -f $(TARGET) $(TARGET_MIC)
