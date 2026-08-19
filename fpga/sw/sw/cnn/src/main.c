#include <stdint.h>
#include <stddef.h>
#include <platform.h>
#include "kprintf.h"

#define REG32(p, i) ((p)[(i) >> 2])

#define SAMPLES_PER_CLASS      500
#define START_SECTOR_WEIGHTS   4096   
#define START_SECTOR_ECG       8192   

#define CNN_CTRL_ADDR          0x06400000L
static volatile uint32_t * const cnn_base = (void *)(CNN_CTRL_ADDR);
static volatile uint32_t * const spi      = (void *)(SPI_CTRL_ADDR);

#define CNN_REG_CONTROL        0x00  
#define CNN_REG_DATA_IN        0x04  
#define CNN_REG_READY          0x08  
#define CNN_REG_CLASSIFICATION 0x0C  
#define CNN_REG_DONE           0x10  

#define CNN_CTRL_RESET         (1 << 0) 
#define CNN_CTRL_START         (1 << 1) 
#define CNN_CTRL_CONFIG        (1 << 2) 
#define CNN_CTRL_WREN          (1 << 3) 

const char *CLASS_NAMES[6] = {
    "Paced beat",                       
    "Atrial premature beat",            
    "Left bundle branch block",         
    "Normal",                           
    "Right bundle branch block",        
    "Premature ventricular contraction" 
};

void print_dec(int val) {
    if (val < 0) { kprintf("-"); val = -val; }
    if (val / 10) print_dec(val / 10);
    char c = '0' + (val % 10);
    kprintf("%c", c);
}

void print_hex32(uint32_t val) {
    char hex_chars[] = "0123456789ABCDEF";
    kprintf("0x");
    for (int i = 7; i >= 0; i--) {
        kprintf("%c", hex_chars[(val >> (i * 4)) & 0x0F]);
    }
}

static inline void delay_cycles(uint32_t cycles) {
    uint32_t start_val;
    __asm__ volatile ("csrr %0, mcycle" : "=r" (start_val));
    while (1) {
        uint32_t current_val;
        __asm__ volatile ("csrr %0, mcycle" : "=r" (current_val));
        if ((current_val - start_val) >= cycles) break;
    }
}

static inline uint8_t sd_spi_xfer(uint8_t d) {
    int32_t r;
    REG32(spi, SPI_REG_TXFIFO) = d;
    do { r = REG32(spi, SPI_REG_RXFIFO); } while (r < 0);
    return (uint8_t)r;
}

int read_sd_sector(uint32_t sector_addr, uint8_t *buffer) {
    REG32(spi, SPI_REG_CSMODE) = SPI_CSMODE_HOLD;
    sd_spi_xfer(0xFF);
    
    sd_spi_xfer(0x51); 
    sd_spi_xfer(sector_addr >> 24); sd_spi_xfer(sector_addr >> 16);
    sd_spi_xfer(sector_addr >> 8);  sd_spi_xfer(sector_addr);
    sd_spi_xfer(0xFF);

    unsigned long n = 50000;
    while ((sd_spi_xfer(0xFF) & 0x80) != 0) if (--n == 0) { REG32(spi, SPI_REG_CSMODE) = SPI_CSMODE_AUTO; return -1; }
    n = 50000;
    while (sd_spi_xfer(0xFF) != 0xFE) if (--n == 0) { REG32(spi, SPI_REG_CSMODE) = SPI_CSMODE_AUTO; return -2; }
    for (int i = 0; i < 512; i++) buffer[i] = sd_spi_xfer(0xFF);
    sd_spi_xfer(0xFF); sd_spi_xfer(0xFF);

    REG32(spi, SPI_REG_CSMODE) = SPI_CSMODE_AUTO;
    for(int d = 0; d < 8; d++) sd_spi_xfer(0xFF);
    return 0;
}

int main(int hartid, char **argv) {
    REG32(uart, UART_REG_TXCTRL) = UART_TXEN;

    kprintf("\r\n========================================================\r\n");
    kprintf(" CNN SYSTEM DEBUG: PREDICTION MAPPING TRACING\r\n");
    kprintf("========================================================\r\n");

    // 1. HARDWARE RESET
    REG32(cnn_base, CNN_REG_CONTROL) = CNN_CTRL_RESET;
    delay_cycles(100);
    REG32(cnn_base, CNN_REG_CONTROL) = 0;
    delay_cycles(100);

    // 2. PRELOAD WEIGHTS T? SD CARD
    uint32_t weight_buffer_32[512];
    uint8_t *weight_buffer_8 = (uint8_t*)weight_buffer_32;

    for (int s = 0; s < 4; s++) {
        read_sd_sector(START_SECTOR_WEIGHTS + s, &weight_buffer_8[s * 512]);
    }

    REG32(cnn_base, CNN_REG_CONTROL) = CNN_CTRL_CONFIG;

    for (int w_idx = 0; w_idx < 458; w_idx++) {
        REG32(cnn_base, CNN_REG_DATA_IN) = weight_buffer_32[w_idx] & 0x000FFFFF;
        REG32(cnn_base, CNN_REG_CONTROL) = CNN_CTRL_CONFIG | CNN_CTRL_WREN;
        REG32(cnn_base, CNN_REG_CONTROL) = CNN_CTRL_CONFIG;
    }
    
    REG32(cnn_base, CNN_REG_CONTROL) = 0;

    kprintf("[TB] SUCCESS: 458 Weights loaded into SRAM.\r\n\r\n");

    uint32_t class_pass_count[6] = {0};
    uint32_t total_pass = 0;

    for (int class_idx = 0; class_idx < 6; class_idx++) {
        uint32_t pass_count = 0;
        uint32_t fail_count = 0;

        kprintf("[TB] EVALUATING CLASS "); print_dec(class_idx); kprintf(": ");
        kprintf((char*)CLASS_NAMES[class_idx]);
        kprintf("\r\n[TB] --------------------------------------------------------\r\n");

        for (int s_idx = 0; s_idx < SAMPLES_PER_CLASS; s_idx++) {
            
            uint32_t global_sample_idx = (class_idx * SAMPLES_PER_CLASS) + s_idx;
            uint32_t target_sector = START_SECTOR_ECG + global_sample_idx;

            uint8_t sector_buffer[512];
            if (read_sd_sector(target_sector, sector_buffer) != 0) continue;

            // STAGE 1: RESET & START
            REG32(cnn_base, CNN_REG_CONTROL) = CNN_CTRL_RESET;
            delay_cycles(20);
            REG32(cnn_base, CNN_REG_CONTROL) = CNN_CTRL_START;

            // STAGE 2: DELAY 28 CYCLES
            delay_cycles(28);

            // Polling ready_for_data == 1
            while ((REG32(cnn_base, CNN_REG_READY) & 0x01) == 0);

            for (int p = 0; p < 3; p++) { 
                REG32(cnn_base, CNN_REG_DATA_IN) = 0; 
                REG32(cnn_base, CNN_REG_CONTROL) = CNN_CTRL_START | CNN_CTRL_WREN;
                REG32(cnn_base, CNN_REG_CONTROL) = CNN_CTRL_START;
            }

            for (int i = 0; i < 200; i++) {
                uint32_t sample_data = ((uint32_t)sector_buffer[i]) & 0x000000FF; 
                REG32(cnn_base, CNN_REG_DATA_IN) = sample_data;
                
                REG32(cnn_base, CNN_REG_CONTROL) = CNN_CTRL_START | CNN_CTRL_WREN;
                REG32(cnn_base, CNN_REG_CONTROL) = CNN_CTRL_START;
            }

            for (int p = 0; p < 3; p++) { 
                REG32(cnn_base, CNN_REG_DATA_IN) = 0; 
                REG32(cnn_base, CNN_REG_CONTROL) = CNN_CTRL_START | CNN_CTRL_WREN;
                REG32(cnn_base, CNN_REG_CONTROL) = CNN_CTRL_START;
            }

            REG32(cnn_base, CNN_REG_CONTROL) = CNN_CTRL_START | CNN_CTRL_WREN;
            REG32(cnn_base, CNN_REG_CONTROL) = CNN_CTRL_START;

            // STAGE 4: POLLING DONE_ALL
            uint32_t timeout_done = 0;
            while ((REG32(cnn_base, CNN_REG_DONE) & 0x01) == 0) {
                if (++timeout_done > 100000) break;
            }

            uint32_t raw_class = REG32(cnn_base, CNN_REG_CLASSIFICATION);
            uint8_t predicted_label = (uint8_t)(raw_class & 0x07);

            REG32(cnn_base, CNN_REG_CONTROL) = 0; // Clear START

            if (s_idx < 5) {
                kprintf("[PREDICT] Expected Class = "); print_dec(class_idx);
                kprintf(" | Got HW Class = "); print_dec(predicted_label);
                if (predicted_label == class_idx) kprintf(" -> PASS\r\n");
                else kprintf(" -> FAIL (Mismatched)\r\n");
            }

            if (predicted_label == class_idx) {
                pass_count++;
            } else {
                fail_count++;
            }
        }

        class_pass_count[class_idx] = pass_count;
        total_pass += pass_count;

        uint32_t class_acc = (pass_count * 100) / SAMPLES_PER_CLASS;
        kprintf("[TB] Class "); print_dec(class_idx);
        kprintf(" Result: PASS="); print_dec(pass_count);
        kprintf(", FAIL="); print_dec(fail_count);
        kprintf(", ACCURACY="); print_dec(class_acc); kprintf("%%\r\n");
        kprintf("--------------------------------------------------------\r\n\r\n");
    }

    kprintf("[TB] ========================================================\r\n");
    kprintf("[TB] FINAL SYSTEM REPORT:\r\n");
    for (int c = 0; c < 6; c++) {
        uint32_t acc = (class_pass_count[c] * 100) / SAMPLES_PER_CLASS;
        kprintf("Class "); print_dec(c); kprintf(": ");
        print_dec(acc); kprintf("%% | ");
        kprintf((char*)CLASS_NAMES[c]);
        kprintf("\r\n");
    }
    kprintf("[TB] --------------------------------------------------------\r\n");
    uint32_t total_accuracy = (total_pass * 100) / (SAMPLES_PER_CLASS * 6);
    kprintf("[TB] AVERAGE ACCURACY (6 CLASSES): "); print_dec(total_accuracy); kprintf("%%\r\n");
    kprintf("[TB] ========================================================\r\n");

    while(1);
    return 0;
}
