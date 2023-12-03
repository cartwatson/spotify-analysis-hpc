#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

std::vector<double*> parseCSV(int maxLines = -1)
{
    std::string fn = "src/data/tracks_features.csv";
    std::ifstream file(fn);
    std::string line;
    int lineCount = 0;
    std::getline(file, line); // Skip header

    std::vector<double*> data;
    std::stringstream ss;
    std::string cell;
    while (std::getline(file, line) && (maxLines == -1 || lineCount < maxLines))
    {
        ss.str(line);
        if (lineCount % 100000 == 0)
            std::cout << "Parsed " << lineCount << " lines" << std::endl;

        std::vector<std::string> row;
        while (std::getline(ss, cell, ','))
            row.push_back(cell);

        double* features = new double[15];
        for (size_t i = row.size() - 15; i < row.size(); i++)
            features[i - (row.size() - 15)] = std::stod(row[i]);

        data.push_back(features);
        lineCount++;
        ss.clear();
    }
    file.close();
    std::cout << "Parsed " << lineCount << " lines" << std::endl;
    return data;
}
