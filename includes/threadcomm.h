
#ifndef THREADCOMM_H_
#define THREADCOMM_H_

class comm
{
	public:
	bool loop_wait = true;
	bool viso_wait_flag = true;
	bool loop_write_done = false;
	bool g2o_loop_flag = false;
	bool isamupdating=false;
};
#endif
