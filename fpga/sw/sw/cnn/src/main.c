#include <stdint.h>
#include <platform.h>
#include "kprintf.h"

#define REG32(p, i) ((p)[(i) >> 2])

#define CNN_CTRL_ADDR 0x06400000L
static volatile uint32_t * const cnn_base = (void *)(CNN_CTRL_ADDR);
static volatile uint32_t * const spi      = (void *)(SPI_CTRL_ADDR);

#define CNN_RESET_OFFSET        0x00
#define CNN_CONFIG_MODE_OFFSET  0x04
#define CNN_WEIGHT_VALID_OFFSET 0x08
#define CNN_WEIGHT_DATA_OFFSET  0x0C
#define CNN_ECG_VALID_OFFSET    0x10
#define CNN_ECG_DATA_OFFSET     0x14
#define CNN_START_INFER_OFFSET  0x18
#define CNN_CLASS_OFFSET        0x1C
#define CNN_DONE_OFFSET         0x20

#define SAMPLES_PER_CLASS       500   // 500 samples * 6 classes = 3000 samples
#define NUM_CLASSES             6
#define START_SECTOR_WEIGHTS    4096  
#define START_SECTOR_ECG        8192  

// Bảng phân bổ sector chuẩn xác cho từng class (mỗi class chiếm 500 sectors)
const uint32_t class_base_sectors[NUM_CLASSES] = {
    8192,  // Class 0: Paced
    8692,  // Class 1: Atrial
    9192,  // Class 2: LBBB
    9692,  // Class 3: Normal
    10192, // Class 4: RBBB
    10692  // Class 5: PVC
};

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
    sd_spi_xfer(0xFF);
    return 0;
}

void print_heartbeat_info(uint8_t label) {
    switch(label) {
        case 0: kprintf("Paced"); break;
        case 1: kprintf("Atrial"); break;
        case 2: kprintf("LBBB"); break;
        case 3: kprintf("Normal"); break;
        case 4: kprintf("RBBB"); break;
        default: kprintf("PVC"); break;
    }
}

int main(int hartid, char **argv) {
    REG32(uart, UART_REG_TXCTRL) = UART_TXEN;
    
    kprintf("\n[C-TB] >>> STARTING FULL 6-CLASS CNN TESTBENCH <<< \r\n");

    // Reset CNN ban đầu
    REG32(cnn_base, CNN_RESET_OFFSET) = 1;
    REG32(cnn_base, CNN_RESET_OFFSET) = 0;
    delay_cycles(120);

    // 1. Đọc trọng số từ thẻ SD (4 sectors từ sector 4096 cho 458 weights)
    uint8_t weight_buffer[2048];
    for (int s = 0; s < 4; s++) {
        int err = read_sd_sector(START_SECTOR_WEIGHTS + s, &weight_buffer[s * 512]);
        if (err != 0) {
            kprintf("[ERROR] SD Weight Read Fail at sector %d\r\n", START_SECTOR_WEIGHTS + s);
            while(1);
        }
    }
    uint32_t *weights_32 = (uint32_t*)weight_buffer;

    kprintf("[C-TB] Step 1: Loading weights...\r\n");
    REG32(cnn_base, CNN_CONFIG_MODE_OFFSET) = 1;

    for (int w = 0; w < 458; w++) {
        REG32(cnn_base, CNN_WEIGHT_DATA_OFFSET) = weights_32[w] & 0x000FFFFF;
        REG32(cnn_base, CNN_WEIGHT_VALID_OFFSET) = 1;
        REG32(cnn_base, CNN_WEIGHT_VALID_OFFSET) = 0;
    }
    
    REG32(cnn_base, CNN_CONFIG_MODE_OFFSET) = 0;
    kprintf("[C-TB] Weight loading finished.\r\n");
    delay_cycles(20);

    int total_pass_global = 0;
    int total_samples_global = NUM_CLASSES * SAMPLES_PER_CLASS;

    // Duyệt qua từng Class (Từ 0 đến 5)
    for (int class_idx = 0; class_idx < NUM_CLASSES; class_idx++) {
        int pass_count = 0;
        int fail_count = 0;

        kprintf("\r\n[C-TB] ===== EVALUATING CLASS %d: ", class_idx);
        print_heartbeat_info(class_idx);
        kprintf(" =====\r\n");

        for (int s_idx = 0; s_idx < SAMPLES_PER_CLASS; s_idx++) {
            uint8_t sector_buffer[512];
            
            uint32_t target_sector = class_base_sectors[class_idx] + s_idx;

            int sd_err = read_sd_sector(target_sector, sector_buffer);
            if (sd_err != 0) {
                kprintf("[ERROR] SD read fail at sector %d (Class %d, Sample %d)\r\n", target_sector, class_idx, s_idx);
                continue;
            }

            // Kích hoạt tín hiệu Start Inference
            REG32(cnn_base, CNN_START_INFER_OFFSET) = 1;
            REG32(cnn_base, CNN_START_INFER_OFFSET) = 0;
            delay_cycles(120);

            // Gửi đúng 200 mẫu ECG thực tế (bỏ qua byte đệm thừa)
            for (int i = 0; i < 200; i++) {
                int8_t raw_val = (int8_t)sector_buffer[i];
                uint32_t sample_data = ((uint32_t)((int32_t)raw_val)) & 0xFF;
                REG32(cnn_base, CNN_ECG_DATA_OFFSET) = sample_data;
                REG32(cnn_base, CNN_ECG_VALID_OFFSET) = 1;
                REG32(cnn_base, CNN_ECG_VALID_OFFSET) = 0;
            }

            uint32_t timeout = 0;
            uint32_t done_status = 0;
            while (1) {
                done_status = REG32(cnn_base, CNN_DONE_OFFSET) & 0x1;
                if (done_status == 1) {
                    break;
                }
                timeout++;
                if (timeout > 500000) {
                    kprintf("[ERROR] CNN Inference Timeout! (Class %d, Sample %d)\r\n", class_idx, s_idx);
                    break;
                }
            }
            
            uint32_t classification_result = REG32(cnn_base, CNN_CLASS_OFFSET);
            uint8_t predicted_label = (uint8_t)(classification_result & 0x07);

            if (predicted_label == (uint8_t)class_idx) {
                pass_count++;
            } else {
                fail_count++;
            }
        }

        total_pass_global += pass_count;
        
        int class_acc_int = (pass_count * 100) / SAMPLES_PER_CLASS;
        int class_acc_frac = ((pass_count * 10000) / SAMPLES_PER_CLASS) % 100;
        
        kprintf("[C-TB] Class %d Result: PASS=%d FAIL=%d ACCURACY=%d.%02d%%\r\n", 
                class_idx, pass_count, fail_count, class_acc_int, class_acc_frac);
    }

    int total_acc_int = (total_pass_global * 100) / total_samples_global;
    int total_acc_frac = ((total_pass_global * 10000) / total_samples_global) % 100;

    kprintf("\r\n[C-TB] ========================================================\r\n");
    kprintf("[C-TB] FINAL SYSTEM REPORT (%d samples/class):\r\n", SAMPLES_PER_CLASS);
    kprintf("[C-TB] AVERAGE ACCURACY (6 CLASSES): %d.%02d%%\r\n", total_acc_int, total_acc_frac);
    kprintf("[C-TB] ========================================================\r\n");

    while(1);
    return 0;
}
