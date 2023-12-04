#include <cmath>

struct Song {
    double feature1, feature2, feature3;
    int cluster;
    double minDist;

    Song(double f1, double f2, double f3):
        feature1(f1),
        feature2(f2),
        feature3(f3),

        cluster(-1),
        minDist(__DBL_MAX__)
    {}

    /**
     * Compute the 3D euclidean distance between this song and another
     */
    double distance(Song other)
    {
        double sum = 0.0;
        sum += (feature1 - other.feature1)*(feature1 - other.feature1);
        sum += (feature2 - other.feature2)*(feature2 - other.feature2);
        sum += (feature3 - other.feature3) * (feature3 - other.feature3);
        return sqrt(sum);
    }
};
