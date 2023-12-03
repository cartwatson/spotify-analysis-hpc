#include <cmath>

struct Instance {
    double danceability,
    energy,
    key,
    loudness,
    mode,
    speechiness,
    acousticness,
    instrumentalness,
    liveness,
    valence,
    tempo,
    duration_ms,
    time_signature;

    int cluster;
    double minDist;

    Instance(double* row):
        danceability(row[0]),
        energy(row[1]),
        key(row[2]),
        loudness(row[3]),
        mode(row[4]),
        speechiness(row[5]),
        acousticness(row[6]),
        instrumentalness(row[7]),
        liveness(row[8]),
        valence(row[9]),
        tempo(row[10]),
        duration_ms(row[11]),
        time_signature(row[12]),
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