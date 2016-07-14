
//New multiGPU header file
//Author: Xin Chen
//May 6, 2016
//Novumind Inc.
//Version 1.0
//
//
#ifndef TIMER_HPP_
#define TIMER_HPP_

#include <sys/time.h>


class Timer
{
public:
    Timer();                                    // default constructor
    ~Timer();                                   // default destructor

    void   start();                             // start timer
    void   stop();                              // stop the timer
    double getElapsedTime();                    // get elapsed time in second
    double getElapsedTimeInSec();               // get elapsed time in second (same as getElapsedTime)
    double getElapsedTimeInMilliSec();          // get elapsed time in milli-second
    double getElapsedTimeInMicroSec();          // get elapsed time in micro-second


protected:


private:
    double startTimeInMicroSec;                 // starting time in micro-second
    double endTimeInMicroSec;                   // ending time in micro-second
    int    stopped;                             // stop flag 
    timeval startCount;                        
    timeval endCount;                          
};

#endif // 
