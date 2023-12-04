#include <cmath>

struct Instance {
    double danceability, acousticness, liveness;
    int cluster;
    double minDist;
    // Default constructor
    Instance() : danceability(0), acousticness(0), liveness(0), cluster(-1), minDist(__DBL_MAX__) {}
    
    Instance(double* row):
        danceability(row[0]),
        acousticness(row[6]),
        liveness(row[8]),

        cluster(-1),
        minDist(__DBL_MAX__)
    {}

    /**
     * Compute the 3D euclidean distance between this instance and another
     */
    double distance(Instance other)
    {
        double danceDiff = other.danceability - danceability;
        double acousticDiff = other.acousticness - acousticness;
        double liveDiff = other.liveness - liveness;
        return std::sqrt(danceDiff * danceDiff + acousticDiff * acousticDiff + liveDiff * liveDiff);
    }
};