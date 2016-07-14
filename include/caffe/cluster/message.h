//Message class hear file
//This file is to define variable and functions of Message class
//Author: Xin Chen
//Novumind Inc.
//Version 1.0

#ifndef _MESSAGE_H_
#define _MESSAGE_H_
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctime>
#include "caffe/cluster/typecommon.hpp"
using namespace std;

char* MsgStr(int command);// Obtain the string of command
char* getTime(); //Get current time format year/month/data/hour:minute:second
char* getTime(int id); //Get current time format year/month/data/hour:minute:second and ID

class Message
{
	
public:
	Message( );
	Message(int Command, int iSource, int iTarget, uint iLength, uchar *pParam=NULL);
    //Message(Message &msg);
	virtual ~Message( );

	Message& operator=(const Message& msg);

	void Clear( );
	int& Command( ) { return m_iCommand; }
	int&	Source( ) { return m_iSource; }
	int& Target( ) { return m_iTarget; }
	uint& Length( ) { return m_iLength; }
	uchar* Buffer( uint iLength );
	uchar *Param( ) { return m_pParam; }
	int& iParam1( ) { return m_Param1.m_iParam1; }
	int& iParam2( ) { return m_Param2.m_iParam2; }
	int& iParam3( ) { return m_Param3.m_iParam3; }
	float& fParam1( ) { return m_Param1.m_fParam1; }
	float& fParam2( ) { return m_Param2.m_fParam2; }
	float& fParam3( ) { return m_Param3.m_fParam3; }
	unsigned char * ucParam1( ) { return m_Param1.m_ucParam1; }
	unsigned char * ucParam2( ) { return m_Param2.m_ucParam2; }
	unsigned char * ucParam3( ) { return m_Param3.m_ucParam3; }


private:
	int	m_iCommand;
	int m_iSource;
	int m_iTarget;
	uint m_iLength;
public:
	union {
		int m_iParam1;
		unsigned char m_ucParam1[4];
		float m_fParam1;
	} m_Param1;
	union {
		int m_iParam2;
		unsigned char m_ucParam2[4];
		float m_fParam2;
	} m_Param2;
	union {
		int m_iParam3;
		unsigned char m_ucParam3[4];
		float m_fParam3;
	} m_Param3;

private:
	unsigned char *m_pParam;
};


//******************************************************************/
// Novunet communcation protocol Format
//******************************************************************/
// All command will follow the following format.
// --------------------------------------------------------------
// | 4 byte | 4 byte | 4 byte | 4byte | 4byte | 4byte | 4byte |............
// |Command | Source | Target | Length|Param1 | Param2| Param3|............        
// --------------------------------------------------------------
//
// 100 -200  commands of traning
// 1000-1999 command among nodes
// 2000-2999 command in one node

//  Training commnds
#define NN_FORWARDBACKWARD_START        100   //start to run en epco
//Infiniband is ready to transfer data
#define NN_INFINIBAND_READY				1000  //InfiniBand information ready
#define NN_INFINIBAND_REMINFO           1001  //Remote node InfiniBand information 



#endif
