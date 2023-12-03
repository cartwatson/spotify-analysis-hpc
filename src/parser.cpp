#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

std::vector<std::vector<double>> parseCSV(int maxLines = -1)
{
    std::string fn = "src/data/tracks_features.csv";
    std::ifstream file(fn);
    std::string line;
    int lineCount = 0;
    std::getline(file, line); // Skip header

    std::vector<std::vector<double>> data;
    std::vector<double> features;
    while (std::getline(file, line) && (maxLines == -1 || lineCount < maxLines))
    {
        if (lineCount % 100000 == 0)
            std::cout << "Parsed " << lineCount << " lines" << std::endl;
        std::stringstream ss(line);
        std::string cell;

        std::vector<std::string> row;
        while (std::getline(ss, cell, ','))
            row.push_back(cell);

        // Get last 15 columns
        std::vector<double> features;
        for (size_t i = row.size() - 15; i < row.size(); i++)
            features.push_back(std::stod(row[i]));
        features.pop_back(); features.pop_back(); // Get rid of dates

        data.push_back(features);
        features.clear();
        lineCount++;
    }

    std::cout << "Parsed " << lineCount << " lines" << std::endl;

    return data;
}
