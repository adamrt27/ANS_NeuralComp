#ifndef DECODE_H
#define DECODE_H

#include <stdint.h>
#include <stdlib.h>

// defines on column of the decodeTable
typedef struct decodeTableColumn {
    int x;
    int sym;
    int nb;
    int newX;
} decodeTableColumn;

// defines the entire decodeTable
typedef struct decodeTable {
    decodeTableColumn *table;
    int L;
    int *s_list;
    int *L_s;
    int n_sym;              // number of symbols
} decodeTable;

// used to hold values for decoding process
typedef struct decoder {
    int state;
    uint8_t *bitstream;
    long l_bitstream;           // the length of the bitstream
    int *msg;
    int l_msg;                  // length of message
} decoder;

// Function declarations

// initialize the decode table
decodeTable *initDecodeTable(int L, int *s_list, int *L_s, int n_sym);
// helper function to create the decode table
void createDecodeTable(decodeTable *table, int* symbol_spread);
// displays the decode table
void displayDecodeTable(decodeTable *table);
// helper function to read bits from the bitstream
int readBits(decoder *d, int nb);
// helper function to decode a step
void decodeStep(decoder *d, decodeTable *table);
// function to decode the bitstream
decoder *decode(uint8_t *bitstream, long l_bitstream, decodeTable *table);

#endif // DECODE_H