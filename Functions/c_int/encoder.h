#ifndef ENCODE_H
#define ENCODE_H

#include <stdlib.h>
#include <stdint.h>

// Encoding

// encode table struct
typedef struct encodeTable{
    int L;
    int *s_list;
    int *L_s;
    int *table;
    int n_sym;              // number of symbols

    int *k;
    int *nb;
    int *start;
} encodeTable;

// struct to handle data while encoding
typedef struct encoder{
    int state;
    uint8_t *bitstream;
    long bitstream_capacity;
    long l_bitstream;
    int *msg;
    int l_msg;
    int ind_msg;                // current index in message you are encoding
} encoder;

// Function declarations

// function to sum the array up to an index
int sum_arr_to_ind(int *arr, int n_sym, int ind);
// Function to initialize the encode table
encodeTable *initEncodeTable(int L, int *s_list, int *L_s, int n_sym);
// helper function to create the encode table
void createEncodeTable(encodeTable *table, const int *sym_spread);
// displays the encode table
void displayEncodeTable(encodeTable *table);
// helper function to append bits to the bitstream
int useBits(encoder *e, int nb);
// helper function to append bits to the bitstream
void append_to_bitstream(encoder *e, int bits, int nb);
// helper function to encode a step
void encode_step(encoder *e, encodeTable *table);
// function to encode the message
encoder *encode(int *msg, int l_msg, encodeTable *table);

#endif // ENCODE_H