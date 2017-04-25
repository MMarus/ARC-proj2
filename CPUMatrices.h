//
// Created by archie on 4/24/17.
//

#ifndef PROJ2_CPUMATRICES_H
#define PROJ2_CPUMATRICES_H

#include <vector>
#include <stdexcept>
#include <mpi.h>
#include <string>
using namespace std;

class CPUMatrices {
public:
    double elapsedTime;
    float *globalArray;
    float *globalDomainParams;
    int *globalDomainMap;


    float *localNewData;
    float *localData;
    int *localDomainMap;
    float *localDomainParams;

    CPUMatrices(int cpus, size_t edge, int rank);
    ~CPUMatrices();
    int myRank;
    int myRow;
    int myCol;
    int myLeftNeighbour;
    int myRightNeighbour;
    int myTopNeighbour;
    int myBottomNeighbour;
    size_t edgeSize;
    size_t widthProcs;
    size_t heightProcs;
    size_t widthEdge;
    size_t heightEdge;
    vector<int> sendcounts;
    vector<int> displs;
    int frameSize[2];
    int frameSizeOfLocal[2];
    int tileSize[2];
    int tileSizeOfLocal[2];
    int starts[2];
    int startsOfLocal[2];

    int numOfCpus;

    MPI_Datatype typeFloat, subArrFloatType;
    MPI_Datatype localSubArrFloatType;
    MPI_Datatype typeInt, subArrIntType;
    MPI_Datatype localSubArrIntType;
    MPI_Datatype localRows;
    MPI_Datatype localCols;

    MPI_Request requests[8];
    MPI_Status statuses[8];
    int requestsCounter = 0;

    int tagTopRows = 5;
    int tagBottomRows = 6;
    int tagLeftCols = 7;
    int tagRightCols = 8;

    int haloTopRowStart;
    int haloDataTopRowStart;
    int haloBottomRowStart;
    int haloDataBottomRowStar;

    int haloLeftColStart;
    int haloDataLeftColStart;
    int haloRightColStart;
    int haloDataRighColStart;


    void copyNewToOld();

    void scatter();
    void gather();
    void sendHaloBlocks();
    void recieveHaloBlocks();

    void printMyData();

    void waitHaloBlocks();

    void recieveHaloBlocksDomainParams();

    void sendHaloBlocksDomainParams();
};



#endif //PROJ2_CPUMATRICES_H
