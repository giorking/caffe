//Implementation of high precision clock
//Author: Xin Chen
//June 1, 2016
//Novumind Inc.
//Version 1.0
//
//
#include "cluster/timer.hpp"
#include <stdlib.h>

Timer1::Timer1()
{
    startCount.tv_sec = startCount.tv_usec = 0;
    endCount.tv_sec = endCount.tv_usec = 0;

    stopped = 0;
    startTimeInMicroSec = 0;
    endTimeInMicroSec = 0;
}

Timer1::~Timer1()
{
}

void Timer1::start()
{
    stopped = 0; // reset stop flag
    gettimeofday(&startCount, NULL);
}

void Timer1::stop()
{
    stopped = 1; // set Timer1 stopped flag

    gettimeofday(&endCount, NULL);
}

double Timer1::getElapsedTimeInMicroSec()
{
    if(!stopped)
        gettimeofday(&endCount, NULL);

    startTimeInMicroSec = (startCount.tv_sec *1000000.0) + startCount.tv_usec;
    endTimeInMicroSec = (endCount.tv_sec * 1000000.0) + endCount.tv_usec;

    return endTimeInMicroSec - startTimeInMicroSec;
}



double Timer1::getElapsedTimeInMilliSec()
{
    return this->getElapsedTimeInMicroSec() * 0.001;
}



double Timer1::getElapsedTimeInSec()
{
    return this->getElapsedTimeInMicroSec() * 0.000001;
}

double Timer1::getElapsedTime()
{
    return this->getElapsedTimeInSec();
}
