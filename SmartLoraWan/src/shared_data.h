#ifndef SHARED_DATA_H
#define SHARED_DATA_H

//#include "ADSBPosition.h"  // if needed

extern ADSBPosition position_log[];
extern int position_count;
extern long last_save_timestamp;
extern bool history_dirty;

#endif
