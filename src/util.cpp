#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <iomanip>

int MAX_LINES = 1204025;

std::vector<double*> parseCSV(int maxLines = MAX_LINES)
{
    std::string fn = "src/data/tracks_features.csv";
    std::ifstream file(fn);
    std::string line;
    int lineCount = 0;
    std::getline(file, line); // Skip header

    std::vector<double*> data;
    std::stringstream ss;
    std::string cell;
    
    const int progressBarWidth = 50;  // Adjust the width of progress bar
    int progressBarStep = maxLines / progressBarWidth;
    
    std::cout << "Parsing input file..." << std::endl;
    while (std::getline(file, line) && lineCount < maxLines)
    {
        ss.str(line);
        if (lineCount % progressBarStep == 0)
        {
            std::cout << "[" << std::setw(progressBarWidth * lineCount / maxLines) << std::setfill('=') << '>'
                      << std::setw(progressBarWidth - progressBarWidth * lineCount / maxLines) << ']' 
                      << " " << std::setw(4) << (100 * lineCount / maxLines) << "%\r" << std::flush;
        }

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
    std::cout << "[" << std::setw(progressBarWidth) << std::setfill('=') << '>'
              << "] " << "100%" << std::endl;
    return data;
}


void writeCSV(std::vector<double*>& data, std::string fn, std::string header)
{
    std::ofstream myfile;
    myfile.open(fn);
    myfile << header << std::endl;

    for (double* feat : data)
        myfile << feat[0] << "," << feat[1] << "," << feat[2] << "," << feat[3] << std::endl;

    myfile.close();
}
