#include <fstream>
#include <iostream>

#ifndef FILEWRITER_H_
#define FILEWRITER_H_

class writer
{
	public:
    std::ofstream my_write_file;
    std::string g2o_string;
    int latestkey1, latestkey2;
};
#endif