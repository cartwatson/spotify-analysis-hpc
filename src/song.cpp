struct Song {
    double feature1, feature2, feature3;
    int cluster;

    Song(): feature1(0), feature2(0), feature3(0), cluster(-1) {}

    Song(double f1, double f2, double f3):
        feature1(f1),
        feature2(f2),
        feature3(f3),
        cluster(-1)
    {}

    /**
     * Compute the 3D euclidean distance between this song and another
     */
    double distance(Song other)
    {
        return (feature1 - other.feature1)*(feature1 - other.feature1)+
               (feature2 - other.feature2)*(feature2 - other.feature2)+
               (feature3 - other.feature3)*(feature3 - other.feature3);
    }
};
