#include <cmath>

struct Song {
    double danceability, // 0.0 to 1.0
        energy, // 0.0 to 1.0
        loudness, // In decibels - Needs to be normalized/scaled
        mode, // 1 or 0 (major or minor)
        speechiness, // 0.0 to 1.0 - .66 and above = mostly speech, .33 to .66 = both music and speech, .33 and below = mostly music
        acousticness, // 0.0 to 1.0
        instrumentalness, // 0.0 to 1.0
        liveness, // 0.0 to 1.0
        valence, // 0.0 to 1.0
        tempo, // In BPM - Needs to be normalized/scaled
        duration_ms, // In milliseconds - Needs to be normalized/scaled
        time_signature; // 3 to 7 - Needs to be normalized/scaled
    int key,

        cluster;
    double minDist;

    Song(double* row):
        danceability(row[0]),
        acousticness(row[6]),
        liveness(row[8]),

        cluster(-1),
        minDist(__DBL_MAX__)
    {}

    /**
     * Compute the 3D euclidean distance between this song and another
     */
    double distance(Song other)
    {
        double danceDiff = other.danceability - danceability;
        double acousticDiff = other.acousticness - acousticness;
        double liveDiff = other.liveness - liveness;
        return std::sqrt(danceDiff * danceDiff + acousticDiff * acousticDiff + liveDiff * liveDiff);
    }
};