//`include "neural_weights_lut.v"
//`include "SRAM1RW128x48.v"
//`include "Neural_Weight_Manager.v"
//`include "cla_20bit.v"
//`include "CNN_Global_Controller.v"
// =============================================================================
// Module Name: CNN_System_Top
// Description: Top-level wrapper for 6-layer CNN (ECG Classification)
// =============================================================================
`timescale 1ns/1ps
module CNN_Wrapper (
    input  wire        clk,
    input  wire        rst_n,

    // Weight SRAM load path
    input  wire        config_mode,   // 1 = weight-load mode (passed through to CNN_System_Top)
    input  wire        weight_valid,  // pulse: write weight_data at the current SRAM address
    input  wire [19:0] weight_data,

    // ECG sample input (buffered into ECG_FIFO ahead of start_infer)
    input  wire        ecg_valid,     // pulse: write ecg_data at the current wr_ptr
    input  wire [7:0]  ecg_data,

    // Pulse to begin classifying the 200 buffered ECG samples
    input  wire        start_infer,

    output wire [2:0]  classification,  // holds the last result
    output wire        done             // 1-cycle pulse when a new result is latched
);

    wire        cnn_rst_n;   // separate CNN core reset
    wire        cnn_start;   // CNN start pulse
    wire        cnn_ready;
    wire [2:0]  cnn_class;
    wire        cnn_done;

    wire [7:0]  ecg_wr_ptr;
    wire [7:0]  ecg_rd_ptr;
    wire [7:0]  ecg_rd_data;

    wire ecg_wr_en, ecg_wr_rst, ecg_rd_rst, ecg_rd_en;

    wire [19:0] cpu_data_in;

    CNN_Ctrl_FSM u_ctrl (
        .clk            (clk),
        .rst_n          (rst_n),
        .start_infer    (start_infer),
        .ecg_valid      (ecg_valid),
        .ecg_wr_ptr     (ecg_wr_ptr),
        .ecg_rd_ptr     (ecg_rd_ptr),
        .cnn_ready      (cnn_ready),
        .cnn_class      (cnn_class),
        .cnn_done       (cnn_done),
        .cnn_rst_n      (cnn_rst_n),
        .cnn_start      (cnn_start),
        .ecg_wr_en      (ecg_wr_en),
        .ecg_wr_rst     (ecg_wr_rst),
        .ecg_rd_rst     (ecg_rd_rst),
        .ecg_rd_en      (ecg_rd_en),
        .classification (classification),
        .done           (done)
    );

    ECG_FIFO u_ecg_fifo (
        .clk     (clk),
        .rst_n   (rst_n),
        .wr_en   (ecg_wr_en),
        .wr_data (ecg_data),
        .wr_rst  (ecg_wr_rst),
        .rd_rst  (ecg_rd_rst),
        .rd_en   (ecg_rd_en),
        .wr_ptr  (ecg_wr_ptr),
        .rd_ptr  (ecg_rd_ptr),
        .rd_data (ecg_rd_data)
    );

    CNN_Data_Mux u_data_mux (
        .sel_weight  (config_mode),
        .weight_data (weight_data),
        .sel_ecg     (ecg_rd_en),
        .ecg_data    (ecg_rd_data),
        .cpu_data_in (cpu_data_in)
    );

    CNN_System_Top u_cnn (
        .clk            (clk),
        .rst_n          (cnn_rst_n),
        .start          (cnn_start),
        .config_mode    (config_mode),
        .cpu_data_in    (cpu_data_in),
        .cpu_wr_en      (weight_valid),
        .ready_for_data (cnn_ready),
        .classification (cnn_class),
        .done_all       (cnn_done)
    );

endmodule
module CNN_Ctrl_FSM (
    input  wire        clk,
    input  wire        rst_n,

    input  wire        start_infer,
    input  wire        ecg_valid,
    input  wire [7:0]  ecg_wr_ptr,
    input  wire [7:0]  ecg_rd_ptr,

    input  wire        cnn_ready,
    input  wire [2:0]  cnn_class,
    input  wire        cnn_done,

    output reg          cnn_rst_n,   // separate CNN core reset
    output reg          cnn_start,   // CNN start pulse

    output wire          ecg_wr_en,
    output wire          ecg_wr_rst,
    output wire          ecg_rd_rst,
    output wire          ecg_rd_en,   // also: mux-select ECG data in CNN_Data_Mux

    output reg  [2:0]   classification,  // holds the last result
    output reg          done             // 1-cycle pulse when a new result is latched
);
    localparam S_IDLE       = 4'd0;
    localparam S_INFER_RST  = 4'd1;
    localparam S_RECV_ECG   = 4'd2;
    localparam S_CNN_RESET  = 4'd3;
    localparam S_CNN_START  = 4'd4;
    localparam S_WAIT_READY = 4'd5;
    localparam S_PRE_PAD    = 4'd6;
    localparam S_FEED_DATA  = 4'd7;
    localparam S_POST_PAD   = 4'd8;
    localparam S_WAIT_DONE  = 4'd9;
    localparam S_RESULT     = 4'd10;

    localparam RESET_CYCLES = 8'd100;

    reg [3:0] state;
    reg [7:0] feed_cnt;   // 0..205 for data feeding
    reg [7:0] rst_cnt;    // CNN reset pulse duration counter

    reg       done_captured;  // latched done_all
    reg [2:0] class_latch;    // latched classification

    // Internal state predicates, combined here into plain FIFO commands.
    wire in_recv_ecg    = (state == S_RECV_ECG);
    wire in_wait_ready  = (state == S_WAIT_READY);
    wire in_pre_pad     = (state == S_PRE_PAD);
    wire in_feed_data   = (state == S_FEED_DATA);
    wire pre_pad_rd_rst = in_pre_pad && (feed_cnt == 8'd3);
    wire ecg_ptr_rst    = (state == S_INFER_RST) && (rst_cnt == RESET_CYCLES - 8'd1);

    assign ecg_wr_en  = in_recv_ecg && ecg_valid;
    assign ecg_wr_rst = ecg_ptr_rst;
    assign ecg_rd_rst = ecg_ptr_rst || (in_wait_ready && cnn_ready) || pre_pad_rd_rst;
    assign ecg_rd_en  = in_feed_data;

    always @(posedge clk or negedge cnn_rst_n) begin
        if (!cnn_rst_n) begin
            done_captured <= 1'b0;
            class_latch   <= 3'd0;
        end else begin
            if (cnn_done) begin
                done_captured <= 1'b1;
                class_latch   <= cnn_class;
            end else if (state == S_RESULT) begin
                done_captured <= 1'b0;  // clear after consumption
            end
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state          <= S_IDLE;
            cnn_rst_n      <= 1'b0;
            cnn_start      <= 1'b0;
            feed_cnt       <= 8'd0;
            rst_cnt        <= 8'd0;
            classification <= 3'd0;
            done           <= 1'b0;
        end else begin
            // Default outputs
            cnn_start <= 1'b0;
            done      <= 1'b0;

            case (state)
                S_IDLE: begin
                    cnn_rst_n <= 1'b1;
                    if (start_infer) begin
                        state     <= S_INFER_RST;
                        cnn_rst_n <= 1'b0;   // assert CNN reset now
                        rst_cnt   <= 8'd0;
                    end
                end

                S_INFER_RST: begin
                    cnn_rst_n <= 1'b0;                  // keep CNN in reset
                    rst_cnt   <= rst_cnt + 8'd1;
                    if (rst_cnt == RESET_CYCLES - 8'd1) begin  // 100 cycles done
                        cnn_rst_n  <= 1'b1;             // release reset
                        state      <= S_RECV_ECG;
                        // ecg_wr_ptr / ecg_rd_ptr reset to 0 (ecg_ptr_rst) happens inside ECG_FIFO
                        feed_cnt   <= 8'd0;
                    end
                end

                S_RECV_ECG: begin
                    // ECG byte write into the FIFO happens inside ECG_FIFO;
                    // ecg_wr_ptr is its current value.
                    if (ecg_valid) begin
                        if (ecg_wr_ptr == 8'd199) begin
                            state   <= S_CNN_RESET;
                            rst_cnt <= 8'd0;
                        end
                    end
                end

                S_CNN_RESET: begin
                    cnn_rst_n <= 1'b0;  // Assert reset
                    rst_cnt   <= rst_cnt + 8'd1;
                    if (rst_cnt == 8'd7) begin  // 8 cycles of reset
                        cnn_rst_n <= 1'b1;  // Release reset
                        state     <= S_CNN_START;
                    end
                end

                S_CNN_START: begin
                    cnn_rst_n <= 1'b1;
                    cnn_start <= 1'b1;  // 1-cycle pulse
                    state     <= S_WAIT_READY;
                end

                S_WAIT_READY: begin
                    if (cnn_ready) begin
                        state    <= S_PRE_PAD;
                        feed_cnt <= 8'd1;  // byte 0 already fed in WAIT_READY
                        // ecg_rd_ptr reset to 0 happens inside ECG_FIFO
                    end
                end

                S_PRE_PAD: begin
                    feed_cnt <= feed_cnt + 8'd1;
                    if (feed_cnt == 8'd3) begin  // 4 leading zeros -> ecg0 at ready_edge+5 (matches golden)
                        state    <= S_FEED_DATA;
                        feed_cnt <= 8'd0;
                        // ecg_rd_ptr reset to 0 (pre_pad_rd_rst) happens inside ECG_FIFO
                    end
                end

                S_FEED_DATA: begin
                    // ecg_rd_ptr increment happens inside ECG_FIFO
                    if (ecg_rd_ptr == 8'd199) begin
                        state    <= S_POST_PAD;
                        feed_cnt <= 8'd0;
                    end
                end

                S_POST_PAD: begin
                    feed_cnt <= feed_cnt + 8'd1;
                    if (feed_cnt == 8'd2) begin  // 3 zeros sent
                        state <= S_WAIT_DONE;
                    end
                end

                S_WAIT_DONE: begin
                    if (done_captured) begin
                        state <= S_RESULT;
                    end
                end

                S_RESULT: begin
                    classification <= class_latch;
                    done           <= 1'b1;  // 1-cycle pulse
                    state          <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
module CNN_Data_Mux (
    input  wire        sel_weight,
    input  wire [19:0] weight_data,
    input  wire        sel_ecg,
    input  wire [7:0]  ecg_data,
    output wire [19:0] cpu_data_in
);

    assign cpu_data_in = sel_weight ? weight_data :
                          sel_ecg   ? {12'd0, ecg_data} : 20'd0;

endmodule
module ECG_FIFO (
    input  wire       clk,
    input  wire       rst_n,

    input  wire       wr_en,    // write wr_data at wr_ptr, then wr_ptr++
    input  wire [7:0] wr_data,
    input  wire       wr_rst,   // wr_ptr <= 0

    input  wire       rd_rst,   // rd_ptr <= 0
    input  wire       rd_en,    // rd_ptr++

    output wire [7:0] wr_ptr,
    output wire [7:0] rd_ptr,
    output wire [7:0] rd_data
);

    reg [7:0] mem [0:199];
    reg [7:0] wr_ptr_r;
    reg [7:0] rd_ptr_r;

    assign wr_ptr  = wr_ptr_r;
    assign rd_ptr  = rd_ptr_r;
    assign rd_data = mem[rd_ptr_r];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_ptr_r <= 8'd0;
            rd_ptr_r <= 8'd0;
        end else begin
            if (wr_rst) begin
                wr_ptr_r <= 8'd0;
            end else if (wr_en) begin
                mem[wr_ptr_r] <= wr_data;
                wr_ptr_r      <= wr_ptr_r + 8'd1;
            end

            if (rd_rst) begin
                rd_ptr_r <= 8'd0;
            end else if (rd_en) begin
                rd_ptr_r <= rd_ptr_r + 8'd1;
            end
        end
    end

endmodule


module CNN_System_Top (
    input  wire              clk,
    input  wire              rst_n,
    input  wire              start,          
    
    input  wire              config_mode,    
    input  wire [19:0]       cpu_data_in,    
    input  wire              cpu_wr_en,      

    output wire              ready_for_data, 
    output wire [2:0]        classification, 
    output wire              done_all        
);

    wire [8:0] internal_cpu_addr;

    Internal_Addr_Counter u_addr_counter (
        .clk         (clk),
        .rst_n       (rst_n),
        .config_mode (config_mode),
        .cpu_wr_en   (cpu_wr_en),
        .addr_out    (internal_cpu_addr)
    );

    wire [8:0]  lut_addr;      
    wire [19:0] lut_data;      
    wire [6:0]  rd_addr, wr_addr;
    wire [2:0]  curr_layer;
    wire        select_pp, done_sys;
    
    wire l0_ld, lr_ld, lr_v_in, l0_sel, pp_we, gap_v_in;
    wire d1_ld, d1_v_in, d2_ld, d2_v_in;
    
    wire l1_v_p, lr_v_p, gap_v, d1_v, d2_v;

    wire [11:0] l1_f0, l1_f1, l1_f2, l1_f3;
    wire [11:0] lr_f0, lr_f1, lr_f2, lr_f3;
    wire [11:0] final_conv_f0, final_conv_f1, final_conv_f2, final_conv_f3;
    wire [11:0] pp_out_f0, pp_out_f1, pp_out_f2, pp_out_f3;
    wire [11:0] gated_f0, gated_f1, gated_f2, gated_f3;
    wire [11:0] gap_f0, gap_f1, gap_f2, gap_f3;
    wire [11:0] d1_out [0:7];
    wire [11:0] d2_out [0:5];

    CNN_Global_Controller u_controller (
        .clk              (clk),
        .rst_n            (rst_n),
        .start            (start),
        .lut_addr         (lut_addr),
        .rd_addr          (rd_addr),
        .wr_addr          (wr_addr),
        .select_pp        (select_pp),
        .curr_layer       (curr_layer),
        .ready_out        (ready_for_data),
        .done_system      (done_sys),
        .l1_v_pool        (l1_v_p),
        .lr_v_pool        (lr_v_p),
        .gap_valid        (gap_v),
        .d1_valid         (d1_v),
        .d2_valid         (d2_v),
        .done_argmax      (done_all),
        .l0_load_en       (l0_ld),
        .lr_load_en       (lr_ld),
        .lr_valid_to_conv (lr_v_in),
        .is_layer0_sel    (l0_sel),
        .pp_wr_en         (pp_we),
        .gap_valid_in     (gap_v_in),
        .d1_load_en       (d1_ld),
        .d1_valid_in      (d1_v_in),
        .d2_load_en       (d2_ld),
        .d2_valid_in      (d2_v_in)
    );

    Neural_Weight_Manager u_weight_manager (
        .clk          (clk),
        .rst_n        (rst_n),
        .config_mode  (config_mode),
        .cpu_addr     (internal_cpu_addr), 
        .cpu_data_in  (cpu_data_in),
        .cpu_wr_en    (cpu_wr_en),
        .cnn_addr     (lut_addr),
        .csd_data_out (lut_data)
    );

    Conv_Output_Mux u_conv_mux (
        .is_layer0_sel (l0_sel),
        .l1_f0(l1_f0), .l1_f1(l1_f1), .l1_f2(l1_f2), .l1_f3(l1_f3),
        .lr_f0(lr_f0), .lr_f1(lr_f1), .lr_f2(lr_f2), .lr_f3(lr_f3),
        .out_f0(final_conv_f0), .out_f1(final_conv_f1), 
        .out_f2(final_conv_f2), .out_f3(final_conv_f3)
    );

    Zero_Gating u_gate (
        .clk     (clk),
        .rd_addr (rd_addr),
        .in_f0(pp_out_f0), .in_f1(pp_out_f1), .in_f2(pp_out_f2), .in_f3(pp_out_f3),
        .out_f0(gated_f0), .out_f1(gated_f1), .out_f2(gated_f2), .out_f3(gated_f3)
    );

    Conv1d0_TOP u_layer0 (
        .clk            (clk),
        .rst_n          (rst_n),
        .load_en        (l0_ld),
        .data_in        ($signed(cpu_data_in[7:0])), 
        .weight_in_lut  (lut_data),
        .out_f0(l1_f0), .out_f1(l1_f1), .out_f2(l1_f2), .out_f3(l1_f3),
        .out_valid_pool (l1_v_p)
    );

    Conv1d_Layer2_TOP u_layer_reuse (
        .clk            (clk),
        .rst_n          (rst_n),
        .load_en        (lr_ld),
        .valid_in       (lr_v_in),
        .in_f0(gated_f0), .in_f1(gated_f1), .in_f2(gated_f2), .in_f3(gated_f3),
        .weight_in_lut  (lut_data),
        .out_f0(lr_f0), .out_f1(lr_f1), .out_f2(lr_f2), .out_f3(lr_f3),
        .out_valid_pool (lr_v_p)
    );

    PingPong_Wrapper u_pingpong (
        .clk     (clk),
        .rst_n   (rst_n),
        .select  (select_pp),
        .rd_addr (rd_addr),
        .wr_addr (wr_addr),
        .wr_en   (pp_we),
        .in_f0(final_conv_f0), .in_f1(final_conv_f1), 
        .in_f2(final_conv_f2), .in_f3(final_conv_f3),
        .out_f0(pp_out_f0), .out_f1(pp_out_f1), 
        .out_f2(pp_out_f2), .out_f3(pp_out_f3)
    );

    Global_Avg_Pool u_gap (
        .clk      (clk),
        .rst_n    (rst_n),
        .valid_in (gap_v_in),
        .in_f0(final_conv_f0), .in_f1(final_conv_f1), 
        .in_f2(final_conv_f2), .in_f3(final_conv_f3),
        .gap_f0(gap_f0), .gap_f1(gap_f1), .gap_f2(gap_f2), .gap_f3(gap_f3),
        .gap_valid(gap_v)
    );

    Dense_Layer_TOP u_dense1 (
        .clk           (clk),
        .rst_n         (rst_n),
        .load_en       (d1_ld),
        .valid_in      (d1_v_in),
        .in_f0(gap_f0), .in_f1(gap_f1), .in_f2(gap_f2), .in_f3(gap_f3),
        .weight_in_lut (lut_data),
        .out_valid     (d1_v),
        .out0(d1_out[0]), .out1(d1_out[1]), .out2(d1_out[2]), .out3(d1_out[3]),
        .out4(d1_out[4]), .out5(d1_out[5]), .out6(d1_out[6]), .out7(d1_out[7])
    );

    Dense_Layer_2_TOP u_dense2 (
        .clk           (clk),
        .rst_n         (rst_n),
        .load_en       (d2_ld),
        .valid_in      (d2_v_in),
        .in0(d1_out[0]), .in1(d1_out[1]), .in2(d1_out[2]), .in3(d1_out[3]),
        .in4(d1_out[4]), .in5(d1_out[5]), .in6(d1_out[6]), .in7(d1_out[7]),
        .weight_in_lut (lut_data),
        .out_valid     (d2_v),
        .out0(d2_out[0]), .out1(d2_out[1]), .out2(d2_out[2]), 
        .out3(d2_out[3]), .out4(d2_out[4]), .out5(d2_out[5])
    );

    Argmax_6 u_argmax (
        .clk       (clk),
        .rst_n     (rst_n),
        .valid_in  (d2_v),
        .in0(d2_out[0]), .in1(d2_out[1]), .in2(d2_out[2]), 
        .in3(d2_out[3]), .in4(d2_out[4]), .in5(d2_out[5]),
        .class_out (classification),
        .valid_out (done_all)
    );

endmodule

module Internal_Addr_Counter (
    input  wire       clk,
    input  wire       rst_n,
    input  wire       config_mode,
    input  wire       cpu_wr_en,
    output reg  [8:0] addr_out
);

    always @(posedge clk) begin
        if (!rst_n) begin
            addr_out <= 9'd0;
        end 
        else if (config_mode) begin
            if (cpu_wr_en) begin
                addr_out <= addr_out + 1'b1;
            end
        end 
        else begin
            addr_out <= 9'd0; // Reset v? 0 khi ? ch? d? ch?y CNN
        end
    end

endmodule


module csd_multiplier_core (
    input  wire               clk,        
    input  wire signed [7:0]  data_in,
    input  wire        [19:0] csd_data,
    output reg  signed [15:0] result  
);
    wire [4:0] t0 = csd_data[4:0];
    wire [4:0] t1 = csd_data[9:5];
    wire [4:0] t2 = csd_data[14:10];
    wire [4:0] t3 = csd_data[19:15];

    wire signed [15:0] data_ext = $signed({{8{data_in[7]}}, data_in});

    reg signed [15:0] v0_reg, v1_reg, v2_reg, v3_reg;
    
    wire signed [15:0] sh0 = data_ext <<< t0[3:0];
    wire signed [15:0] sh1 = data_ext <<< t1[3:0];
    wire signed [15:0] sh2 = data_ext <<< t2[3:0];
    wire signed [15:0] sh3 = data_ext <<< t3[3:0];

    wire signed [15:0] sh0_neg = -sh0;
    wire signed [15:0] sh1_neg = -sh1;
    wire signed [15:0] sh2_neg = -sh2;
    wire signed [15:0] sh3_neg = -sh3;

    always @(posedge clk) begin
        v0_reg <= (t0 == 5'b11111) ? 16'sb0 : (t0[4] ? sh0_neg : sh0);
        v1_reg <= (t1 == 5'b11111) ? 16'sb0 : (t1[4] ? sh1_neg : sh1);
        v2_reg <= (t2 == 5'b11111) ? 16'sb0 : (t2[4] ? sh2_neg : sh2);
        v3_reg <= (t3 == 5'b11111) ? 16'sb0 : (t3[4] ? sh3_neg : sh3);
    end

    reg signed [15:0] sum_l0_a_reg, sum_l0_b_reg;

    wire signed [15:0] sum_l0_a = v0_reg + v1_reg;
    wire signed [15:0] sum_l0_b = v2_reg + v3_reg;

    always @(posedge clk) begin
        sum_l0_a_reg <= sum_l0_a;
        sum_l0_b_reg <= sum_l0_b;
    end

    wire signed [15:0] final_res = sum_l0_a_reg + sum_l0_b_reg;

    always @(posedge clk) begin
        result <= final_res; 
    end

endmodule

module pe_direct (
    input  wire                clk,        
    input  wire signed [7:0]   data_in,
    input  wire signed [19:0]  weight_in, 
    output reg  signed [15:0]  p_out      
);

    wire signed [27:0] full_product;
    assign full_product = data_in * weight_in;

    reg signed [27:0] prod_stage1;
    always @(posedge clk) begin
        prod_stage1 <= full_product; 
    end

    reg signed [27:0] prod_stage2;
    always @(posedge clk) begin
        prod_stage2 <= prod_stage1;
    end

    always @(posedge clk) begin
        p_out <= prod_stage2[15:0]; 
    end

endmodule


module Conv1_Top (
    input  wire          clk, rst_n, load_en,
    input  wire signed [7:0] data_in,
    input  wire        [19:0] weight_in_lut,
    output reg           out_valid, 
    output reg  signed [15:0] filter0_out, filter1_out, filter2_out, filter3_out
);

    // ---------------------------------------------------------
    // 1. KH?I QU?N LÝ VALID 
    // ---------------------------------------------------------
    reg [3:0] valid_cnt;
  always @(posedge clk) begin
        if (!rst_n) begin
            valid_cnt <= 0;
            out_valid <= 0;
        end else if (load_en) begin
            valid_cnt <= 0;
            out_valid <= 0;
        end else begin
          if (valid_cnt < 13) begin
                valid_cnt <= valid_cnt + 1'b1;
                out_valid <= 0;
            end else begin
                out_valid <= 1;
            end
        end
    end
    reg signed [7:0] d0, d1, d2, d3, d4, d5, d6;
  always @(posedge clk ) begin
        if (!rst_n) begin
            d0 <= 0; d1 <= 0; d2 <= 0; d3 <= 0; d4 <= 0; d5 <= 0; d6 <= 0;
        end else begin
            if (load_en) begin
                d0 <= 8'd0; d1 <= 8'd0; d2 <= 8'd0; d3 <= 8'd0; d4 <= 8'd0; d5 <= 8'd0; d6 <= 8'd0;
            end else begin
                d0 <= data_in;
                d1 <= d0; d2 <= d1; d3 <= d2; d4 <= d3; d5 <= d4; d6 <= d5;
            end
        end
    end
    reg [19:0] w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, 
               w14, w15, w16, w17, w18, w19, w20, w21, w22, w23, w24, w25, w26, w27;

  always @(posedge clk ) begin
        if (!rst_n) begin
            {w0,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13,w14,w15,w16,w17,w18,w19,w20,w21,w22,w23,w24,w25,w26,w27} <= 0;
        end else if (load_en) begin
            w27 <= weight_in_lut;
            w26 <= w27; w25 <= w26; w24 <= w25; w23 <= w24; w22 <= w23; w21 <= w22; w20 <= w21;
            w19 <= w20; w18 <= w19; w17 <= w18; w16 <= w17; w15 <= w16; w14 <= w15; w13 <= w14;
            w12 <= w13; w11 <= w12; w10 <= w11; w9  <= w10; w8  <= w9;  w7  <= w8;  w6  <= w7;
            w5  <= w6;  w4  <= w5;  w3  <= w4;  w2  <= w3;  w1  <= w2;  w0  <= w1;
        end
    end
    
    wire signed [15:0] p0[0:6], p1[0:6], p2[0:6], p3[0:6];

    // Filter 0
    pe_direct pe0_0 (clk, d6, w0,  p0[0]); pe_direct pe0_1 (clk, d5, w4,  p0[1]); 
    pe_direct pe0_2 (clk, d4, w8,  p0[2]); pe_direct pe0_3 (clk, d3, w12, p0[3]); 
    pe_direct pe0_4 (clk, d2, w16, p0[4]); pe_direct pe0_5 (clk, d1, w20, p0[5]);
    pe_direct pe0_6 (clk, d0, w24, p0[6]);

    // Filter 1
    pe_direct pe1_0 (clk, d6, w1,  p1[0]); pe_direct pe1_1 (clk, d5, w5,  p1[1]); 
    pe_direct pe1_2 (clk, d4, w9,  p1[2]); pe_direct pe1_3 (clk, d3, w13, p1[3]); 
    pe_direct pe1_4 (clk, d2, w17, p1[4]); pe_direct pe1_5 (clk, d1, w21, p1[5]);
    pe_direct pe1_6 (clk, d0, w25, p1[6]);

    // Filter 2
    pe_direct pe2_0 (clk, d6, w2,  p2[0]); pe_direct pe2_1 (clk, d5, w6,  p2[1]); 
    pe_direct pe2_2 (clk, d4, w10, p2[2]); pe_direct pe2_3 (clk, d3, w14, p2[3]); 
    pe_direct pe2_4 (clk, d2, w18, p2[4]); pe_direct pe2_5 (clk, d1, w22, p2[5]);
    pe_direct pe2_6 (clk, d0, w26, p2[6]);

    // Filter 3
    pe_direct pe3_0 (clk, d6, w3,  p3[0]); pe_direct pe3_1 (clk, d5, w7,  p3[1]); 
    pe_direct pe3_2 (clk, d4, w11, p3[2]); pe_direct pe3_3 (clk, d3, w15, p3[3]); 
    pe_direct pe3_4 (clk, d2, w19, p3[4]); pe_direct pe3_5 (clk, d1, w23, p3[5]);
    pe_direct pe3_6 (clk, d0, w27, p3[6]);

    // 3. Thanh ghi bù tr? cho nhánh PE s? 6 (Bù 1 chu k? c?a T?ng c?ng 1)
    reg signed [15:0] p06_delay, p16_delay, p26_delay, p36_delay;
    always @(posedge clk) begin
        p06_delay <= p0[6];
        p16_delay <= p1[6];
        p26_delay <= p2[6];
        p36_delay <= p3[6];
    end


    reg signed [15:0] f0_s1_01, f0_s1_23, f0_s1_45;
    reg signed [15:0] f0_s2_0123, f0_s2_456;

    always @(posedge clk) begin
        // T?ng 1: Th?c hi?n phép c?ng toán t? '+' và luu vào thanh ghi pipeline
        f0_s1_01   <= p0[0] + p0[1];
        f0_s1_23   <= p0[2] + p0[3];
        f0_s1_45   <= p0[4] + p0[5];
        
        // T?ng 2: C?ng k?t qu? t? t?ng 1
        f0_s2_0123 <= f0_s1_01 + f0_s1_23;
        f0_s2_456  <= f0_s1_45 + p06_delay;
        
        // T?ng 3: K?t qu? cu?i cùng du?c gán tr?c ti?p vào output (chuy?n sang d?ng reg)
        filter0_out <= f0_s2_0123 + f0_s2_456;
    end

    // --- CÂY C?NG PIPELINE CHO FILTER 1 ---
    reg signed [15:0] f1_s1_01, f1_s1_23, f1_s1_45;
    reg signed [15:0] f1_s2_0123, f1_s2_456;

    always @(posedge clk) begin
        f1_s1_01   <= p1[0] + p1[1];
        f1_s1_23   <= p1[2] + p1[3];
        f1_s1_45   <= p1[4] + p1[5];
        
        f1_s2_0123 <= f1_s1_01 + f1_s1_23;
        f1_s2_456  <= f1_s1_45 + p16_delay;
        
        filter1_out <= f1_s2_0123 + f1_s2_456;
    end

    // --- CÂY C?NG PIPELINE CHO FILTER 2 ---
    reg signed [15:0] f2_s1_01, f2_s1_23, f2_s1_45;
    reg signed [15:0] f2_s2_0123, f2_s2_456;

    always @(posedge clk) begin
        f2_s1_01   <= p2[0] + p2[1];
        f2_s1_23   <= p2[2] + p2[3];
        f2_s1_45   <= p2[4] + p2[5];
        
        f2_s2_0123 <= f2_s1_01 + f2_s1_23;
        f2_s2_456  <= f2_s1_45 + p26_delay;
        
        filter2_out <= f2_s2_0123 + f2_s2_456;
    end

    // --- CÂY C?NG PIPELINE CHO FILTER 3 ---
    reg signed [15:0] f3_s1_01, f3_s1_23, f3_s1_45;
    reg signed [15:0] f3_s2_0123, f3_s2_456;

    always @(posedge clk) begin
        f3_s1_01   <= p3[0] + p3[1];
        f3_s1_23   <= p3[2] + p3[3];
        f3_s1_45   <= p3[4] + p3[5];
        
        f3_s2_0123 <= f3_s1_01 + f3_s1_23;
        f3_s2_456  <= f3_s1_45 + p36_delay;
        
        filter3_out <= f3_s2_0123 + f3_s2_456;
    end

endmodule

// --- Kh?i ReLU x? lý cho 1 Filter (Ngõ ra 12-bit) dã t?i uu cho FPGA ---
module ReLU_Unit (
    input  wire signed [15:0] data_in,    
    input  wire signed [15:0] bias_in,    
    output wire signed [11:0] data_out    
);

    wire signed [15:0] sum_round = data_in + bias_in + 16'sd16;

    // D?ch bit s? h?c (S? d?ng toán t? <<< ho?c >>> v?i ki?u d? li?u signed)
    wire signed [15:0] s_shifted = sum_round >>> 5;

    assign data_out = (s_shifted[15] == 1'b1) ? 12'sd0 :         // N?u âm -> 0
                      (s_shifted > 16'sd2047) ? 12'sd2047 :      // N?u vu?t ngu?ng 12-bit -> Max 12-bit
                      s_shifted[11:0];                           // Còn l?i l?y 12 bit th?p

endmodule


module MaxPool_Unit (
    input  clk, rst_n,
    input  valid_in,
    input  signed [11:0] data_in,    
    output reg signed [11:0] data_out, 
    output reg valid_out
);
    reg [1:0] state;
    reg signed [11:0] max_val;
    
    // --- THÊM B? Ð?M N?I B? ---
    reg [6:0] count_out; // Ð?m d?n 66 m?u d?u ra

  always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state     <= 2'd0;
            max_val   <= 12'sh800;   
            data_out  <= 12'd0;
            valid_out <= 1'b0;
            count_out <= 7'd0;       // Reset b? d?m
        end else if (valid_in) begin
            case (state)
                2'd0: begin
                    max_val   <= data_in;
                    state     <= 2'd1;
                    valid_out <= 1'b0;
                end
                2'd1: begin
                    if (data_in > max_val) max_val <= data_in;
                    state     <= 2'd2;
                    valid_out <= 1'b0;
                end
                2'd2: begin
                    // Tính giá tr? Max cu?i cùng
                    if (data_in > max_val) data_out <= data_in;
                    else                   data_out <= max_val;
                    
                    // --- LOGIC KHÓA VALID_OUT T?I ÐÂY ---
                  if (count_out < 7'd69) begin
                        valid_out <= 1'b1;
                        count_out <= count_out + 7'd1; // Tang khi xu?t m?u thành công
                    end else begin
                        valid_out <= 1'b0; // Ðã d? 66 m?u, "dóng c?a" luôn
                    end
                    
                    state     <= 2'd0;
                    max_val   <= 12'sh800; 
                end
                default: state <= 2'd0;
            endcase
        end else begin
            valid_out <= 1'b0;
        end
    end
endmodule



module Conv1d0_TOP (
    input  clk,
    input  rst_n,
    input  load_en,          // Tín hi?u n?p tr?ng s?
    input  signed [7:0] data_in,
    input  [19:0] weight_in_lut,
    
    // Ð?u ra sau khi dã x? lý xong t?ng 1 (Conv + ReLU + Pool)
  	output [11:0] out_f0, out_f1, out_f2, out_f3,
    output       out_valid_pool  // Tín hi?u báo d? li?u Pool dã s?n sàng
);

    // --- 1. Tín hi?u k?t n?i trung gian ---
    wire out_valid_conv;
    wire signed [15:0] f0_raw, f1_raw, f2_raw, f3_raw;
  	wire [11:0] f0_relu, f1_relu, f2_relu, f3_relu;

    // --- 2. Kh?i Convolution 1D (7-tap, 4 Filter) ---
    // S? d?ng cây c?ng CLA 16-bit n?i b?
    Conv1_Top u_conv (
        .clk(clk),
        .rst_n(rst_n),
        .load_en(load_en),
        .data_in(data_in),
        .weight_in_lut(weight_in_lut),
        .out_valid(out_valid_conv),
        .filter0_out(f0_raw),
        .filter1_out(f1_raw),
        .filter2_out(f2_raw),
        .filter3_out(f3_raw)
    );

    // --- 3. Kh?i ReLU (G?m Bias, Rounding, Shift, ReLU) ---
    // M?i kh?i th?c hi?n: (raw + bias + 16) >>> 5 và c?t ngu?ng ReLU
    ReLU_Unit u_relu0 (.data_in(f0_raw), .bias_in(16'd0), .data_out(f0_relu));
    ReLU_Unit u_relu1 (.data_in(f1_raw), .bias_in(16'd0), .data_out(f1_relu));
    ReLU_Unit u_relu2 (.data_in(f2_raw), .bias_in(16'd0), .data_out(f2_relu));
    ReLU_Unit u_relu3 (.data_in(f3_raw), .bias_in(16'd0), .data_out(f3_relu));

    // --- 4. Kh?i Max-Pooling 
    MaxPool_Unit u_pool0 (
        .clk(clk), .rst_n(rst_n),
        .valid_in(out_valid_conv),
        .data_in(f0_relu),
        .data_out(out_f0),
        .valid_out(out_valid_pool) 
    );

    MaxPool_Unit u_pool1 (
        .clk(clk), .rst_n(rst_n),
        .valid_in(out_valid_conv),
        .data_in(f1_relu),
        .data_out(out_f1),
        .valid_out()
    );

    MaxPool_Unit u_pool2 (
        .clk(clk), .rst_n(rst_n),
        .valid_in(out_valid_conv),
        .data_in(f2_relu),
        .data_out(out_f2),
        .valid_out()
    );

    MaxPool_Unit u_pool3 (
        .clk(clk), .rst_n(rst_n),
        .valid_in(out_valid_conv),
        .data_in(f3_relu),
        .data_out(out_f3),
        .valid_out()
    );

endmodule

///////////////l?p conv th? 2

module csd_multiplier_core_conv2 (
    input  wire               clk,        // Chân clock
    input  wire signed [11:0] data_in,   
    input  wire        [19:0] csd_data,  
    output reg  signed [19:0] result      // S? d?ng output reg tr?c ti?p
);
    // 1. Trích xu?t các tham s? (Logic t? h?p)
    wire [4:0] t0 = csd_data[4:0];
    wire [4:0] t1 = csd_data[9:5];
    wire [4:0] t2 = csd_data[14:10];
    wire [4:0] t3 = csd_data[19:15];

    // Sign-extension t? 12-bit lên 20-bit (Ép ki?u signed d? chu?n hóa d? li?u)
    wire signed [19:0] data_ext = $signed({{8{data_in[11]}}, data_in});

    // --- STAGE 1: Shifting and Negating ---
    reg signed [19:0] v0_reg, v1_reg, v2_reg, v3_reg;
    
    // S? d?ng toán t? d?ch bit s? h?c <<< d? b?o toàn bit d?u
    wire signed [19:0] sh0 = data_ext <<< t0[3:0];
    wire signed [19:0] sh1 = data_ext <<< t1[3:0];
    wire signed [19:0] sh2 = data_ext <<< t2[3:0];
    wire signed [19:0] sh3 = data_ext <<< t3[3:0];

    // Thay th? 'negator_20bit' b?ng toán t? tr? '-' chuyên d?ng c?a FPGA
    wire signed [19:0] sh0_neg = -sh0;
    wire signed [19:0] sh1_neg = -sh1;
    wire signed [19:0] sh2_neg = -sh2;
    wire signed [19:0] sh3_neg = -sh3;

    always @(posedge clk) begin
        v0_reg <= (t0 == 5'b11111) ? 20'sb0 : (t0[4] ? sh0_neg : sh0);
        v1_reg <= (t1 == 5'b11111) ? 20'sb0 : (t1[4] ? sh1_neg : sh1);
        v2_reg <= (t2 == 5'b11111) ? 20'sb0 : (t2[4] ? sh2_neg : sh2);
        v3_reg <= (t3 == 5'b11111) ? 20'sb0 : (t3[4] ? sh3_neg : sh3);
    end

    // --- STAGE 2: First Level Adder Tree ---
    reg signed [19:0] sum_l0_a_reg, sum_l0_b_reg;

    // Thay th? 'cla_20bit' b?ng toán t? c?ng '+'
    wire signed [19:0] sum_l0_a = v0_reg + v1_reg;
    wire signed [19:0] sum_l0_b = v2_reg + v3_reg;

    always @(posedge clk) begin
        sum_l0_a_reg <= sum_l0_a;
        sum_l0_b_reg <= sum_l0_b;
    end

    // --- STAGE 3: Final Adder (Ch?t vào ngõ ra) ---
    // Thay th? kh?i 'cla_20bit' cu?i cùng b?ng toán t? '+'
    wire signed [19:0] final_res = sum_l0_a_reg + sum_l0_b_reg;

    always @(posedge clk) begin
        result <= final_res;
    end

endmodule

module pe_direct_conv2 (
    input  wire                clk,         // Chân clock d?ng b?
    input  wire signed [11:0]  data_in,
    input  wire signed [19:0]  weight_csd,  // Tr?ng s? d?ng signed s? nguyên th?c s?
    output reg  signed [19:0]  p_out        // Ch?t ngõ ra ? STAGE 3
);

    wire signed [31:0] full_product;
    assign full_product = data_in * weight_csd;

    reg signed [31:0] prod_stage1;
    always @(posedge clk) begin
        prod_stage1 <= full_product; 
    end

    reg signed [31:0] prod_stage2;
    always @(posedge clk) begin
        prod_stage2 <= prod_stage1;
    end

    always @(posedge clk) begin
        p_out <= prod_stage2[19:0]; 
    end

endmodule

module Conv2_Top (
    input  clk, rst_n, load_en,
    input  valid_in, // valid_out t\u1eeb Layer 1
  	input  signed [11:0] in_f0, in_f1, in_f2, in_f3,
    input  [19:0] weight_in_lut,
    output reg out_valid, 
  	output signed [19:0] filter0_out, filter1_out, filter2_out, filter3_out
);

    // ---------------------------------------------------------
    // 1. KH\u1ed0I QU\u1ea2N LÝ VALID
    // ---------------------------------------------------------
  reg [3:0] valid_cnt;
  always @(posedge clk ) begin
        if (!rst_n) begin
            valid_cnt <= 0;
            out_valid <= 0;
        end else if (load_en) begin
            valid_cnt <= 0;
            out_valid <= 0;
        end else if (valid_in && !load_en) begin
          if (valid_cnt < 14) begin
                valid_cnt <= valid_cnt + 1;
                out_valid <= 0;
            end else begin
                out_valid <= 1;
            end
        end
    end

    // ---------------------------------------------------------
    // 2. KH\u1ed0I SHIFT REGISTER D\u1eee LI\u1ec6U (4 Hàng x 7 Taps)
    // ---------------------------------------------------------
  	reg signed [11:0] df0[0:6], df1[0:6], df2[0:6], df3[0:6];
    always @(posedge clk ) begin
        if (!rst_n) begin
            df0[0]<=0; df0[1]<=0; df0[2]<=0; df0[3]<=0; df0[4]<=0; df0[5]<=0; df0[6]<=0;
            df1[0]<=0; df1[1]<=0; df1[2]<=0; df1[3]<=0; df1[4]<=0; df1[5]<=0; df1[6]<=0;
            df2[0]<=0; df2[1]<=0; df2[2]<=0; df2[3]<=0; df2[4]<=0; df2[5]<=0; df2[6]<=0;
            df3[0]<=0; df3[1]<=0; df3[2]<=0; df3[3]<=0; df3[4]<=0; df3[5]<=0; df3[6]<=0;
        end else if (valid_in) begin
            df0[0]<=in_f0; df0[1]<=df0[0]; df0[2]<=df0[1]; df0[3]<=df0[2]; df0[4]<=df0[3]; df0[5]<=df0[4]; df0[6]<=df0[5];
            df1[0]<=in_f1; df1[1]<=df1[0]; df1[2]<=df1[1]; df1[3]<=df1[2]; df1[4]<=df1[3]; df1[5]<=df1[4]; df1[6]<=df1[5];
            df2[0]<=in_f2; df2[1]<=df2[0]; df2[2]<=df2[1]; df2[3]<=df2[2]; df2[4]<=df2[3]; df2[5]<=df2[4]; df2[6]<=df2[5];
            df3[0]<=in_f3; df3[1]<=df3[0]; df3[2]<=df3[1]; df3[3]<=df3[2]; df3[4]<=df3[3]; df3[5]<=df3[4]; df3[6]<=df3[5];
        end
    end

    // ---------------------------------------------------------
    // 3. KH\u1ed0I QU\u1ea2N LÝ TR\u1eccNG S\u1ed0 (Scan-chain 112 ph\u1ea7n t\u1eed)
    // ---------------------------------------------------------
    reg [19:0] w[0:111];
  always @(posedge clk ) begin
        if (!rst_n) begin
            // Kh\u1edfi t\u1ea1o w[0] \u0111\u1ebfn w[111] = 0
            {w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8],w[9],w[10],w[11],w[12],w[13],w[14],w[15],w[16],w[17],w[18],w[19],w[20],w[21],w[22],w[23],w[24],w[25],w[26],w[27],w[28],w[29],w[30],w[31],w[32],w[33],w[34],w[35],w[36],w[37],w[38],w[39],w[40],w[41],w[42],w[43],w[44],w[45],w[46],w[47],w[48],w[49],w[50],w[51],w[52],w[53],w[54],w[55],w[56],w[57],w[58],w[59],w[60],w[61],w[62],w[63],w[64],w[65],w[66],w[67],w[68],w[69],w[70],w[71],w[72],w[73],w[74],w[75],w[76],w[77],w[78],w[79],w[80],w[81],w[82],w[83],w[84],w[85],w[86],w[87],w[88],w[89],w[90],w[91],w[92],w[93],w[94],w[95],w[96],w[97],w[98],w[99],w[100],w[101],w[102],w[103],w[104],w[105],w[106],w[107],w[108],w[109],w[110],w[111]} <= 0;
        end else if (load_en) begin
            w[111] <= weight_in_lut;
            w[110] <= w[111]; w[109] <= w[110]; w[108] <= w[109]; w[107] <= w[108]; w[106] <= w[107]; w[105] <= w[106]; w[104] <= w[105];
            w[103] <= w[104]; w[102] <= w[103]; w[101] <= w[102]; w[100] <= w[101]; w[99] <= w[100];  w[98] <= w[99];   w[97] <= w[98]; w[96] <= w[97];
            w[95] <= w[96]; w[94] <= w[95]; w[93] <= w[94]; w[92] <= w[93]; w[91] <= w[92]; w[90] <= w[91]; w[89] <= w[90]; w[88] <= w[89];
            w[87] <= w[88]; w[86] <= w[87]; w[85] <= w[86]; w[84] <= w[85]; w[83] <= w[84]; w[82] <= w[83]; w[81] <= w[82]; w[80] <= w[81];
            w[79] <= w[80]; w[78] <= w[79]; w[77] <= w[78]; w[76] <= w[77]; w[75] <= w[76]; w[74] <= w[75]; w[73] <= w[74]; w[72] <= w[73];
            w[71] <= w[72]; w[70] <= w[71]; w[69] <= w[70]; w[68] <= w[69]; w[67] <= w[68]; w[66] <= w[67]; w[65] <= w[66]; w[64] <= w[65];
            w[63] <= w[64]; w[62] <= w[63]; w[61] <= w[62]; w[60] <= w[61]; w[59] <= w[60]; w[58] <= w[59]; w[57] <= w[58]; w[56] <= w[57];
            w[55] <= w[56]; w[54] <= w[55]; w[53] <= w[54]; w[52] <= w[53]; w[51] <= w[52]; w[50] <= w[51]; w[49] <= w[50]; w[48] <= w[49];
            w[47] <= w[48]; w[46] <= w[47]; w[45] <= w[46]; w[44] <= w[45]; w[43] <= w[44]; w[42] <= w[43]; w[41] <= w[42]; w[40] <= w[41];
            w[39] <= w[40]; w[38] <= w[39]; w[37] <= w[38]; w[36] <= w[37]; w[35] <= w[36]; w[34] <= w[35]; w[33] <= w[34]; w[32] <= w[33];
            w[31] <= w[32]; w[30] <= w[31]; w[29] <= w[30]; w[28] <= w[29]; w[27] <= w[28]; w[26] <= w[27]; w[25] <= w[26]; w[24] <= w[25];
            w[23] <= w[24]; w[22] <= w[23]; w[21] <= w[22]; w[20] <= w[21]; w[19] <= w[20]; w[18] <= w[19]; w[17] <= w[18]; w[16] <= w[17];
            w[15] <= w[16]; w[14] <= w[15]; w[13] <= w[14]; w[12] <= w[13]; w[11] <= w[12]; w[10] <= w[11]; w[9]  <= w[10]; w[8]  <= w[9];
            w[7]  <= w[8];  w[6]  <= w[7];  w[5]  <= w[6];  w[4]  <= w[5];  w[3]  <= w[4];  w[2]  <= w[3];  w[1]  <= w[2];  w[0]  <= w[1];
        end
    end

    wire signed [19:0] p0[0:27];

    // 2. Kh?i t?o các PE (K?t n?i df0..df3, tr?ng s? w và chân clk)
    // Tap 0
    pe_direct_conv2 pe0_k0_c0 (clk, df0[6], w[0],   p0[0]);  pe_direct_conv2 pe0_k0_c1 (clk, df1[6], w[4],   p0[1]);  
    pe_direct_conv2 pe0_k0_c2 (clk, df2[6], w[8],   p0[2]);  pe_direct_conv2 pe0_k0_c3 (clk, df3[6], w[12],  p0[3]);
    // Tap 1
    pe_direct_conv2 pe0_k1_c0 (clk, df0[5], w[16],  p0[4]);  pe_direct_conv2 pe0_k1_c1 (clk, df1[5], w[20],  p0[5]);  
    pe_direct_conv2 pe0_k1_c2 (clk, df2[5], w[24],  p0[6]);  pe_direct_conv2 pe0_k1_c3 (clk, df3[5], w[28],  p0[7]);
    // Tap 2
    pe_direct_conv2 pe0_k2_c0 (clk, df0[4], w[32],  p0[8]);  pe_direct_conv2 pe0_k2_c1 (clk, df1[4], w[36],  p0[9]);  
    pe_direct_conv2 pe0_k2_c2 (clk, df2[4], w[40],  p0[10]); pe_direct_conv2 pe0_k2_c3 (clk, df3[4], w[44],  p0[11]);
    // Tap 3
    pe_direct_conv2 pe0_k3_c0 (clk, df0[3], w[48],  p0[12]); pe_direct_conv2 pe0_k3_c1 (clk, df1[3], w[52],  p0[13]);  
    pe_direct_conv2 pe0_k3_c2 (clk, df2[3], w[56],  p0[14]); pe_direct_conv2 pe0_k3_c3 (clk, df3[3], w[60],  p0[15]);
    // Tap 4
    pe_direct_conv2 pe0_k4_c0 (clk, df0[2], w[64],  p0[16]); pe_direct_conv2 pe0_k4_c1 (clk, df1[2], w[68],  p0[17]);  
    pe_direct_conv2 pe0_k4_c2 (clk, df2[2], w[72],  p0[18]); pe_direct_conv2 pe0_k4_c3 (clk, df3[2], w[76],  p0[19]);
    // Tap 5
    pe_direct_conv2 pe0_k5_c0 (clk, df0[1], w[80],  p0[20]); pe_direct_conv2 pe0_k5_c1 (clk, df1[1], w[84],  p0[21]);  
    pe_direct_conv2 pe0_k5_c2 (clk, df2[1], w[88],  p0[22]); pe_direct_conv2 pe0_k5_c3 (clk, df3[1], w[92],  p0[23]);
    // Tap 6
    pe_direct_conv2 pe0_k6_c0 (clk, df0[0], w[96],  p0[24]); pe_direct_conv2 pe0_k6_c1 (clk, df1[0], w[100], p0[25]);  
    pe_direct_conv2 pe0_k6_c2 (clk, df2[0], w[104], p0[26]); pe_direct_conv2 pe0_k6_c3 (clk, df3[0], w[108], p0[27]);


    // ---------------------------------------------------------
    // 5. KH\u1ed0I TÍNH TOÁN FILTER 1 (Cout=1)
    // Ánh x\u1ea1 \u0111\u1ecba ch\u1ec9 LUT 29, 33, 37... vào PE (w[1], w[5], w[9]...)
    // ---------------------------------------------------------
    
    // 1. Khai báo wire k?t n?i tr?c ti?p t? ngõ ra PE
    wire signed [19:0] p1[0:27];

    // 2. Kh?i t?o các PE cho Filter 1 (Tr?ng s? w[1], w[5], w[9]... và chân clk)
    // Tap 0
    pe_direct_conv2 pe1_k0_c0 (clk, df0[6], w[1],   p1[0]);  pe_direct_conv2 pe1_k0_c1 (clk, df1[6], w[5],   p1[1]);  
    pe_direct_conv2 pe1_k0_c2 (clk, df2[6], w[9],   p1[2]);  pe_direct_conv2 pe1_k0_c3 (clk, df3[6], w[13],  p1[3]);
    // Tap 1
    pe_direct_conv2 pe1_k1_c0 (clk, df0[5], w[17],  p1[4]);  pe_direct_conv2 pe1_k1_c1 (clk, df1[5], w[21],  p1[5]);  
    pe_direct_conv2 pe1_k1_c2 (clk, df2[5], w[25],  p1[6]);  pe_direct_conv2 pe1_k1_c3 (clk, df3[5], w[29],  p1[7]);
    // Tap 2
    pe_direct_conv2 pe1_k2_c0 (clk, df0[4], w[33],  p1[8]);  pe_direct_conv2 pe1_k2_c1 (clk, df1[4], w[37],  p1[9]);  
    pe_direct_conv2 pe1_k2_c2 (clk, df2[4], w[41],  p1[10]); pe_direct_conv2 pe1_k2_c3 (clk, df3[4], w[45],  p1[11]);
    // Tap 3
    pe_direct_conv2 pe1_k3_c0 (clk, df0[3], w[49],  p1[12]); pe_direct_conv2 pe1_k3_c1 (clk, df1[3], w[53],  p1[13]);  
    pe_direct_conv2 pe1_k3_c2 (clk, df2[3], w[57],  p1[14]); pe_direct_conv2 pe1_k3_c3 (clk, df3[3], w[61],  p1[15]);
    // Tap 4
    pe_direct_conv2 pe1_k4_c0 (clk, df0[2], w[65],  p1[16]); pe_direct_conv2 pe1_k4_c1 (clk, df1[2], w[69],  p1[17]);  
    pe_direct_conv2 pe1_k4_c2 (clk, df2[2], w[73],  p1[18]); pe_direct_conv2 pe1_k4_c3 (clk, df3[2], w[77],  p1[19]);
    // Tap 5
    pe_direct_conv2 pe1_k5_c0 (clk, df0[1], w[81],  p1[20]); pe_direct_conv2 pe1_k5_c1 (clk, df1[1], w[85],  p1[21]);  
    pe_direct_conv2 pe1_k5_c2 (clk, df2[1], w[89],  p1[22]); pe_direct_conv2 pe1_k5_c3 (clk, df3[1], w[93],  p1[23]);
    // Tap 6
    pe_direct_conv2 pe1_k6_c0 (clk, df0[0], w[97],  p1[24]); pe_direct_conv2 pe1_k6_c1 (clk, df1[0], w[101], p1[25]);  
    pe_direct_conv2 pe1_k6_c2 (clk, df2[0], w[105], p1[26]); pe_direct_conv2 pe1_k6_c3 (clk, df3[0], w[109], p1[27]);
  
  // ---------------------------------------------------------
    // 6. KH\u1ed0I TÍNH TOÁN FILTER 2 (Cout=2)
    // Ánh x\u1ea1 \u0111\u1ecba ch\u1ec9 LUT 30, 34, 38... vào PE (w[2], w[6], w[10]...)
    // ---------------------------------------------------------
    
    // 1. Khai báo wire k?t n?i tr?c ti?p t? ngõ ra PE
    wire signed [19:0] p2[0:27];

    // 2. Kh?i t?o các PE cho Filter 2 (Tr?ng s? w[2], w[6], w[10]... và chân clk)
    // Tap 0
    pe_direct_conv2 pe2_k0_c0 (clk, df0[6], w[2],   p2[0]);  pe_direct_conv2 pe2_k0_c1 (clk, df1[6], w[6],   p2[1]);  
    pe_direct_conv2 pe2_k0_c2 (clk, df2[6], w[10],  p2[2]);  pe_direct_conv2 pe2_k0_c3 (clk, df3[6], w[14],  p2[3]);
    // Tap 1
    pe_direct_conv2 pe2_k1_c0 (clk, df0[5], w[18],  p2[4]);  pe_direct_conv2 pe2_k1_c1 (clk, df1[5], w[22],  p2[5]);  
    pe_direct_conv2 pe2_k1_c2 (clk, df2[5], w[26],  p2[6]);  pe_direct_conv2 pe2_k1_c3 (clk, df3[5], w[30],  p2[7]);
    // Tap 2
    pe_direct_conv2 pe2_k2_c0 (clk, df0[4], w[34],  p2[8]);  pe_direct_conv2 pe2_k2_c1 (clk, df1[4], w[38],  p2[9]);  
    pe_direct_conv2 pe2_k2_c2 (clk, df2[4], w[42],  p2[10]); pe_direct_conv2 pe2_k2_c3 (clk, df3[4], w[46],  p2[11]);
    // Tap 3
    pe_direct_conv2 pe2_k3_c0 (clk, df0[3], w[50],  p2[12]); pe_direct_conv2 pe2_k3_c1 (clk, df1[3], w[54],  p2[13]);  
    pe_direct_conv2 pe2_k3_c2 (clk, df2[3], w[58],  p2[14]); pe_direct_conv2 pe2_k3_c3 (clk, df3[3], w[62],  p2[15]);
    // Tap 4
    pe_direct_conv2 pe2_k4_c0 (clk, df0[2], w[66],  p2[16]); pe_direct_conv2 pe2_k4_c1 (clk, df1[2], w[70],  p2[17]);  
    pe_direct_conv2 pe2_k4_c2 (clk, df2[2], w[74],  p2[18]); pe_direct_conv2 pe2_k4_c3 (clk, df3[2], w[78],  p2[19]);
    // Tap 5
    pe_direct_conv2 pe2_k5_c0 (clk, df0[1], w[82],  p2[20]); pe_direct_conv2 pe2_k5_c1 (clk, df1[1], w[86],  p2[21]);  
    pe_direct_conv2 pe2_k5_c2 (clk, df2[1], w[90],  p2[22]); pe_direct_conv2 pe2_k5_c3 (clk, df3[1], w[94],  p2[23]);
    // Tap 6
    pe_direct_conv2 pe2_k6_c0 (clk, df0[0], w[98],  p2[24]); pe_direct_conv2 pe2_k6_c1 (clk, df1[0], w[102], p2[25]);  
    pe_direct_conv2 pe2_k6_c2 (clk, df2[0], w[106], p2[26]); pe_direct_conv2 pe2_k6_c3 (clk, df3[0], w[110], p2[27]);
  
  	// ---------------------------------------------------------
    // 7. KH\u1ed0I TÍNH TOÁN FILTER 3 (Cout=3)
    // Ánh x\u1ea1 \u0111\u1ecba ch\u1ec9 LUT 31, 35, 39... vào PE (w[3], w[7], w[11]...)
    // ---------------------------------------------------------
    
    // 1. Khai báo wire k?t n?i tr?c ti?p t? ngõ ra PE
    wire signed [19:0] p3[0:27];

    // 2. Kh?i t?o các PE cho Filter 3 (Tr?ng s? w[3], w[7], w[11]... và chân clk)
    // Tap 0
    pe_direct_conv2 pe3_k0_c0 (clk, df0[6], w[3],   p3[0]);  pe_direct_conv2 pe3_k0_c1 (clk, df1[6], w[7],   p3[1]);  
    pe_direct_conv2 pe3_k0_c2 (clk, df2[6], w[11],  p3[2]);  pe_direct_conv2 pe3_k0_c3 (clk, df3[6], w[15],  p3[3]);
    // Tap 1
    pe_direct_conv2 pe3_k1_c0 (clk, df0[5], w[19],  p3[4]);  pe_direct_conv2 pe3_k1_c1 (clk, df1[5], w[23],  p3[5]);  
    pe_direct_conv2 pe3_k1_c2 (clk, df2[5], w[27],  p3[6]);  pe_direct_conv2 pe3_k1_c3 (clk, df3[5], w[31],  p3[7]);
    // Tap 2
    pe_direct_conv2 pe3_k2_c0 (clk, df0[4], w[35],  p3[8]);  pe_direct_conv2 pe3_k2_c1 (clk, df1[4], w[39],  p3[9]);  
    pe_direct_conv2 pe3_k2_c2 (clk, df2[4], w[43],  p3[10]); pe_direct_conv2 pe3_k2_c3 (clk, df3[4], w[47],  p3[11]);
    // Tap 3
    pe_direct_conv2 pe3_k3_c0 (clk, df0[3], w[51],  p3[12]); pe_direct_conv2 pe3_k3_c1 (clk, df1[3], w[55],  p3[13]);  
    pe_direct_conv2 pe3_k3_c2 (clk, df2[3], w[59],  p3[14]); pe_direct_conv2 pe3_k3_c3 (clk, df3[3], w[63],  p3[15]);
    // Tap 4
    pe_direct_conv2 pe3_k4_c0 (clk, df0[2], w[67],  p3[16]); pe_direct_conv2 pe3_k4_c1 (clk, df1[2], w[71],  p3[17]);  
    pe_direct_conv2 pe3_k4_c2 (clk, df2[2], w[75],  p3[18]); pe_direct_conv2 pe3_k4_c3 (clk, df3[2], w[79],  p3[19]);
    // Tap 5
    pe_direct_conv2 pe3_k5_c0 (clk, df0[1], w[83],  p3[20]); pe_direct_conv2 pe3_k5_c1 (clk, df1[1], w[87],  p3[21]);  
    pe_direct_conv2 pe3_k5_c2 (clk, df2[1], w[91],  p3[22]); pe_direct_conv2 pe3_k5_c3 (clk, df3[1], w[95],  p3[23]);
    // Tap 6
    pe_direct_conv2 pe3_k6_c0 (clk, df0[0], w[99],  p3[24]); pe_direct_conv2 pe3_k6_c1 (clk, df1[0], w[103], p3[25]);  
  	pe_direct_conv2 pe3_k6_c2 (clk, df2[0], w[107], p3[26]); pe_direct_conv2 pe3_k6_c3 (clk, df3[0], w[111], p3[27]);

    // =========================================================
    // CÂY C?NG PIPELINE CHO CÁC FILTER (KHÔNG DÙNG VÒNG L?P)
    // =========================================================
    
    // --- Khai báo các thanh ghi t?ng trung gian ---
    reg signed [19:0] f0_t1[0:13], f1_t1[0:13], f2_t1[0:13], f3_t1[0:13];
    reg signed [19:0] f0_t2[0:6],  f1_t2[0:6],  f2_t2[0:6],  f3_t2[0:6];
    reg signed [19:0] f0_t3[0:3],  f1_t3[0:3],  f2_t3[0:3],  f3_t3[0:3];
    reg signed [19:0] f0_t4_0, f0_t4_1, f1_t4_0, f1_t4_1, f2_t4_0, f2_t4_1, f3_t4_0, f3_t4_1;
    reg signed [19:0] f0_out_reg, f1_out_reg, f2_out_reg, f3_out_reg;

    always @(posedge clk) begin
        // --- T?NG 1: 14 B? c?ng (Tr? 1 clk so v?i PE) ---
        // Filter 0
        f0_t1[0]  <= p0[0]  + p0[1];  f0_t1[1]  <= p0[2]  + p0[3];
        f0_t1[2]  <= p0[4]  + p0[5];  f0_t1[3]  <= p0[6]  + p0[7];
        f0_t1[4]  <= p0[8]  + p0[9];  f0_t1[5]  <= p0[10] + p0[11];
        f0_t1[6]  <= p0[12] + p0[13]; f0_t1[7]  <= p0[14] + p0[15];
        f0_t1[8]  <= p0[16] + p0[17]; f0_t1[9]  <= p0[18] + p0[19];
        f0_t1[10] <= p0[20] + p0[21]; f0_t1[11] <= p0[22] + p0[23];
        f0_t1[12] <= p0[24] + p0[25]; f0_t1[13] <= p0[26] + p0[27];
        // Filter 1
        f1_t1[0]  <= p1[0]  + p1[1];  f1_t1[1]  <= p1[2]  + p1[3];
        f1_t1[2]  <= p1[4]  + p1[5];  f1_t1[3]  <= p1[6]  + p1[7];
        f1_t1[4]  <= p1[8]  + p1[9];  f1_t1[5]  <= p1[10] + p1[11];
        f1_t1[6]  <= p1[12] + p1[13]; f1_t1[7]  <= p1[14] + p1[15];
        f1_t1[8]  <= p1[16] + p1[17]; f1_t1[9]  <= p1[18] + p1[19];
        f1_t1[10] <= p1[20] + p1[21]; f1_t1[11] <= p1[22] + p1[23];
        f1_t1[12] <= p1[24] + p1[25]; f1_t1[13] <= p1[26] + p1[27];
        // Filter 2
        f2_t1[0]  <= p2[0]  + p2[1];  f2_t1[1]  <= p2[2]  + p2[3];
        f2_t1[2]  <= p2[4]  + p2[5];  f2_t1[3]  <= p2[6]  + p2[7];
        f2_t1[4]  <= p2[8]  + p2[9];  f2_t1[5]  <= p2[10] + p2[11];
        f2_t1[6]  <= p2[12] + p2[13]; f2_t1[7]  <= p2[14] + p2[15];
        f2_t1[8]  <= p2[16] + p2[17]; f2_t1[9]  <= p2[18] + p2[19];
        f2_t1[10] <= p2[20] + p2[21]; f2_t1[11] <= p2[22] + p2[23];
        f2_t1[12] <= p2[24] + p2[25]; f2_t1[13] <= p2[26] + p2[27];
        // Filter 3
        f3_t1[0]  <= p3[0]  + p3[1];  f3_t1[1]  <= p3[2]  + p3[3];
        f3_t1[2]  <= p3[4]  + p3[5];  f3_t1[3]  <= p3[6]  + p3[7];
        f3_t1[4]  <= p3[8]  + p3[9];  f3_t1[5]  <= p3[10] + p3[11];
        f3_t1[6]  <= p3[12] + p3[13]; f3_t1[7]  <= p3[14] + p3[15];
        f3_t1[8]  <= p3[16] + p3[17]; f3_t1[9]  <= p3[18] + p3[19];
        f3_t1[10] <= p3[20] + p3[21]; f3_t1[11] <= p3[22] + p3[23];
        f3_t1[12] <= p3[24] + p3[25]; f3_t1[13] <= p3[26] + p3[27];

        // --- T?NG 2: 7 B? c?ng (Tr? 2 clk so v?i PE) ---
        // Filter 0
        f0_t2[0] <= f0_t1[0]  + f0_t1[1];  f0_t2[1] <= f0_t1[2]  + f0_t1[3];
        f0_t2[2] <= f0_t1[4]  + f0_t1[5];  f0_t2[3] <= f0_t1[6]  + f0_t1[7];
        f0_t2[4] <= f0_t1[8]  + f0_t1[9];  f0_t2[5] <= f0_t1[10] + f0_t1[11];
        f0_t2[6] <= f0_t1[12] + f0_t1[13];
        // Filter 1
        f1_t2[0] <= f1_t1[0]  + f1_t1[1];  f1_t2[1] <= f1_t1[2]  + f1_t1[3];
        f1_t2[2] <= f1_t1[4]  + f1_t1[5];  f1_t2[3] <= f1_t1[6]  + f1_t1[7];
        f1_t2[4] <= f1_t1[8]  + f1_t1[9];  f1_t2[5] <= f1_t1[10] + f1_t1[11];
        f1_t2[6] <= f1_t1[12] + f1_t1[13];
        // Filter 2
        f2_t2[0] <= f2_t1[0]  + f2_t1[1];  f2_t2[1] <= f2_t1[2]  + f2_t1[3];
        f2_t2[2] <= f2_t1[4]  + f2_t1[5];  f2_t2[3] <= f2_t1[6]  + f2_t1[7];
        f2_t2[4] <= f2_t1[8]  + f2_t1[9];  f2_t2[5] <= f2_t1[10] + f2_t1[11];
        f2_t2[6] <= f2_t1[12] + f2_t1[13];
        // Filter 3
        f3_t2[0] <= f3_t1[0]  + f3_t1[1];  f3_t2[1] <= f3_t1[2]  + f3_t1[3];
        f3_t2[2] <= f3_t1[4]  + f3_t1[5];  f3_t2[3] <= f3_t1[6]  + f3_t1[7];
        f3_t2[4] <= f3_t1[8]  + f3_t1[9];  f3_t2[5] <= f3_t1[10] + f3_t1[11];
        f3_t2[6] <= f3_t1[12] + f3_t1[13];

        // --- T?NG 3: 4 Ð?u ra (Tr? 3 clk so v?i PE - Ðã g?p nhánh bù tr? l?) ---
        // Filter 0
        f0_t3[0] <= f0_t2[0] + f0_t2[1]; f0_t3[1] <= f0_t2[2] + f0_t2[3];
        f0_t3[2] <= f0_t2[4] + f0_t2[5]; f0_t3[3] <= f0_t2[6]; // Ði th?ng qua reg d? bù tr?
        // Filter 1
        f1_t3[0] <= f1_t2[0] + f1_t2[1]; f1_t3[1] <= f1_t2[2] + f1_t2[3];
        f1_t3[2] <= f1_t2[4] + f1_t2[5]; f1_t3[3] <= f1_t2[6];
        // Filter 2
        f2_t3[0] <= f2_t2[0] + f2_t2[1]; f2_t3[1] <= f2_t2[2] + f2_t2[3];
        f2_t3[2] <= f2_t2[4] + f2_t2[5]; f2_t3[3] <= f2_t2[6];
        // Filter 3
        f3_t3[0] <= f3_t2[0] + f3_t2[1]; f3_t3[1] <= f3_t2[2] + f3_t2[3];
        f3_t3[2] <= f3_t2[4] + f3_t2[5]; f3_t3[3] <= f3_t2[6];

        // --- T?NG 4: 2 B? c?ng (Tr? 4 clk so v?i PE) ---
        f0_t4_0 <= f0_t3[0] + f0_t3[1]; f0_t4_1 <= f0_t3[2] + f0_t3[3];
        f1_t4_0 <= f1_t3[0] + f1_t3[1]; f1_t4_1 <= f1_t3[2] + f1_t3[3];
        f2_t4_0 <= f2_t3[0] + f2_t3[1]; f2_t4_1 <= f2_t3[2] + f2_t3[3];
        f3_t4_0 <= f3_t3[0] + f3_t3[1]; f3_t4_1 <= f3_t3[2] + f3_t3[3];

        // --- T?NG CU?I: Ch?t ngõ ra (Tr? dúng 5 clk so v?i PE) ---
        f0_out_reg <= f0_t4_0 + f0_t4_1;
        f1_out_reg <= f1_t4_0 + f1_t4_1;
        f2_out_reg <= f2_t4_0 + f2_t4_1;
        f3_out_reg <= f3_t4_0 + f3_t4_1;
    end

    // Gán liên t?c ra các Port ngõ ra d?ng wire c?a Top module
    assign filter0_out = f0_out_reg;
    assign filter1_out = f1_out_reg;
    assign filter2_out = f2_out_reg;
    assign filter3_out = f3_out_reg;

endmodule

// --- Kh?i ReLU x? lý cho L?p 2 (Nén t? 20-bit signed v? 12-bit signed) ---
module ReLU_Unit_conv2 (
    input  wire signed [19:0] data_in,    // Ð?u vào t? Conv2_Top (20-bit)
    input  wire signed [19:0] bias_in,    // Bias tuong ?ng (20-bit)
    output wire signed [11:0] data_out    // Ngõ ra 12-bit cho MaxPool/Layer ti?p theo
);

    // G?p phép c?ng bias và c?ng làm tròn (Rounding) thành m?t bi?u th?c.
    // Trình t?ng h?p FPGA s? t? d?ng gom c?m này t?i uu ph?n c?ng.
    wire signed [19:0] sum_round = data_in + bias_in + 20'sd16;

    // D?ch bit s? h?c d? scale down 5 bit (B?o toàn bit d?u)
    wire signed [19:0] s_shifted = sum_round >>> 5;

    // ReLU + Saturation (Bão hòa v? 12-bit signed: 0 d?n 2047)
    // Ð?ng b? toàn b? h?ng s? sang ki?u s? có d?u 'signed' (sd)
    assign data_out = (s_shifted[19] == 1'b1) ? 12'sd0 :         // N?u âm -> 0 (ReLU)
                      (s_shifted > 20'sd2047) ? 12'sd2047 :      // N?u vu?t ngu?ng -> Max 12-bit signed
                      s_shifted[11:0];                           // Còn l?i l?y 12 bit th?p

endmodule

module MaxPool_Unit_conv2 (
    input  clk, rst_n,
    input  valid_in,
    input  signed [11:0] data_in,    // Ð?u vào 12-bit t? ReLU
    output reg signed [11:0] data_out, 
    output reg valid_out
);
    reg [1:0] state;
    reg signed [11:0] max_val;

  always @(posedge clk) begin
        if (!rst_n) begin
            state     <= 2'd0;
            max_val   <= 12'sh800;   // Giá tr? âm nh? nh?t 12-bit (-2048)
            data_out  <= 12'd0;
            valid_out <= 1'b0;
        end else if (valid_in) begin
            case (state)
                2'd0: begin
                    max_val   <= data_in;
                    state     <= 2'd1;
                    valid_out <= 1'b0;
                end
                2'd1: begin
                    if (data_in > max_val) max_val <= data_in;
                    state     <= 2'd2;
                    valid_out <= 1'b0;
                end
                2'd2: begin
                    // So sánh n?t m?u th? 3 và xu?t k?t qu?
                    if (data_in > max_val) data_out <= data_in;
                    else                   data_out <= max_val;
                    
                    valid_out <= 1'b1;
                    state     <= 2'd0;
                end
                default: state <= 2'd0;
            endcase
        end else begin
            valid_out <= 1'b0;
        end
    end
endmodule

module Conv1d_Layer2_TOP (
    input  clk,
    input  rst_n,
    input  load_en,           
    input  valid_in,          
    input  signed [11:0] in_f0, in_f1, in_f2, in_f3, 
    input  [19:0] weight_in_lut,
    
    output [11:0] out_f0, out_f1, out_f2, out_f3,
    output        out_valid_pool  
);

    // --- 1. Tín hi?u k?t n?i trung gian ---
    wire out_valid_conv;
    wire signed [19:0] f0_raw, f1_raw, f2_raw, f3_raw; 
    
    // --- 2. PIPELINE STAGE 1: Sau kh?i Conv ---
    // M?c tiêu: Ng?t du?ng logic t? b? nhân CSD/B? c?ng sang ReLU
    reg  signed [19:0] f0_raw_reg, f1_raw_reg, f2_raw_reg, f3_raw_reg;
    reg                valid_conv_reg;

    Conv2_Top u_conv2 (
        .clk(clk), .rst_n(rst_n), .load_en(load_en), .valid_in(valid_in),
        .in_f0(in_f0), .in_f1(in_f1), .in_f2(in_f2), .in_f3(in_f3),
        .weight_in_lut(weight_in_lut),
        .out_valid(out_valid_conv),
        .filter0_out(f0_raw), .filter1_out(f1_raw), .filter2_out(f2_raw), .filter3_out(f3_raw)
    );
  
  always @(posedge clk ) begin
        if (!rst_n) begin
            f0_raw_reg <= 20'd0; f1_raw_reg <= 20'd0;
            f2_raw_reg <= 20'd0; f3_raw_reg <= 20'd0;
            valid_conv_reg <= 1'b0;
        end else begin
            valid_conv_reg <= out_valid_conv;
            if (out_valid_conv) begin
                f0_raw_reg <= f0_raw; f1_raw_reg <= f1_raw;
                f2_raw_reg <= f2_raw; f3_raw_reg <= f3_raw;
            end
        end
    end

    // --- 3. Kh?i ReLU_Unit_conv2 (L?y d? li?u t? Stage 1) ---
    wire [11:0] f0_relu, f1_relu, f2_relu, f3_relu;

    ReLU_Unit_conv2 u_relu_f0 (.data_in(f0_raw_reg), .bias_in(20'd0), .data_out(f0_relu));
    ReLU_Unit_conv2 u_relu_f1 (.data_in(f1_raw_reg), .bias_in(20'd0), .data_out(f1_relu));
    ReLU_Unit_conv2 u_relu_f2 (.data_in(f2_raw_reg), .bias_in(20'd0), .data_out(f2_relu));
    ReLU_Unit_conv2 u_relu_f3 (.data_in(f3_raw_reg), .bias_in(20'd0), .data_out(f3_relu));

    // --- 4. PIPELINE STAGE 2: Sau kh?i ReLU ---
    // M?c tiêu: Ng?t du?ng logic t? b? c?ng trong ReLU sang b? so sánh c?a MaxPool
    reg [11:0] f0_relu_reg, f1_relu_reg, f2_relu_reg, f3_relu_reg;
    reg        valid_relu_reg;

  always @(posedge clk ) begin
        if (!rst_n) begin
            f0_relu_reg <= 12'd0; f1_relu_reg <= 12'd0;
            f2_relu_reg <= 12'd0; f3_relu_reg <= 12'd0;
            valid_relu_reg <= 1'b0;
        end else begin
            valid_relu_reg <= valid_conv_reg;
            if (valid_conv_reg) begin
                f0_relu_reg <= f0_relu; f1_relu_reg <= f1_relu;
                f2_relu_reg <= f2_relu; f3_relu_reg <= f3_relu;
            end
        end
    end

    // --- 5. Kh?i Max-Pooling Layer 2 (L?y d? li?u t? Stage 2) ---
    MaxPool_Unit_conv2 u_pool_f0 (
        .clk(clk), .rst_n(rst_n),
        .valid_in(valid_relu_reg),
        .data_in(f0_relu_reg),
        .data_out(out_f0),
        .valid_out(out_valid_pool)
    );

    MaxPool_Unit_conv2 u_pool_f1 (
        .clk(clk), .rst_n(rst_n),
        .valid_in(valid_relu_reg),
        .data_in(f1_relu_reg),
        .data_out(out_f1),
        .valid_out() 
    );

    MaxPool_Unit_conv2 u_pool_f2 (
        .clk(clk), .rst_n(rst_n),
        .valid_in(valid_relu_reg),
        .data_in(f2_relu_reg),
        .data_out(out_f2),
        .valid_out()
    );

    MaxPool_Unit_conv2 u_pool_f3 (
        .clk(clk), .rst_n(rst_n),
        .valid_in(valid_relu_reg),
        .data_in(f3_relu_reg),
        .data_out(out_f3),
        .valid_out()
    );

endmodule


module Global_Avg_Pool (
    input  wire        clk, rst_n,
    input  wire        valid_in,          // Tín hi?u valid t? MaxPool Layer 4
    input  wire signed [11:0] in_f0, in_f1, in_f2, in_f3,
    output reg  signed [11:0] gap_f0, gap_f1, gap_f2, gap_f3,
    output reg         gap_valid
);

    // --- 1. Thanh ghi luu tr? m?u th? nh?t ---
    reg signed [11:0] reg_f0, reg_f1, reg_f2, reg_f3;
    reg state; // 0: Ð?i m?u 1, 1: Ð?i m?u 2 và tính toán

    // --- 2. Ép ki?u t? d?ng (Sign-extension) t? 12-bit sang 16-bit signed ---
    wire signed [15:0] ext_in_f0  = $signed(in_f0);
    wire signed [15:0] ext_reg_f0 = $signed(reg_f0);
    
    wire signed [15:0] ext_in_f1  = $signed(in_f1);
    wire signed [15:0] ext_reg_f1 = $signed(reg_f1);
    
    wire signed [15:0] ext_in_f2  = $signed(in_f2);
    wire signed [15:0] ext_reg_f2 = $signed(reg_f2);
    
    wire signed [15:0] ext_in_f3  = $signed(in_f3);
    wire signed [15:0] ext_reg_f3 = $signed(reg_f3);

    // --- 3. Th?c hi?n phép c?ng s? h?c thu?n túy (Thay cho b? c?ng CLA) ---
    wire signed [15:0] sum_f0 = ext_reg_f0 + ext_in_f0;
    wire signed [15:0] sum_f1 = ext_reg_f1 + ext_in_f1;
    wire signed [15:0] sum_f2 = ext_reg_f2 + ext_in_f2;
    wire signed [15:0] sum_f3 = ext_reg_f3 + ext_in_f3;

    // --- 4. Logic di?u khi?n FSM ---
  always @(posedge clk ) begin
        if (!rst_n) begin
            state <= 1'b0;
            gap_valid <= 1'b0;
            reg_f0 <= 12'sb0; reg_f1 <= 12'sb0; reg_f2 <= 12'sb0; reg_f3 <= 12'sb0;
            gap_f0 <= 12'sb0; gap_f1 <= 12'sb0; gap_f2 <= 12'sb0; gap_f3 <= 12'sb0;
        end else if (valid_in) begin
            if (state == 1'b0) begin
                // Luu m?u d?u tiên (x0)
                reg_f0 <= in_f0;
                reg_f1 <= in_f1;
                reg_f2 <= in_f2;
                reg_f3 <= in_f3;
                state  <= 1'b1;
                gap_valid <= 1'b0;
            end else begin
                // Ðã có m?u th? hai (x1), ti?n hành chia dôi l?y trung bình (sum >>> 1)
                // Ép ki?u c?t bit ngu?c v? 12-bit signed ? ngõ ra
                gap_f0 <= sum_f0 >>> 1;
                gap_f1 <= sum_f1 >>> 1;
                gap_f2 <= sum_f2 >>> 1;
                gap_f3 <= sum_f3 >>> 1;
                
                gap_valid <= 1'b1;
                state <= 1'b0; // Quay l?i ch? c?p d? li?u m?i
            end
        end else begin
            gap_valid <= 1'b0;
        end
    end

endmodule


module dense_neuron (
    input  wire        clk,
    input  wire signed [11:0] x0, x1, x2, x3,
    input  wire        [19:0] w0, w1, w2, w3,
    input  wire signed [19:0] bias,
    output wire signed [19:0] neuron_out
);

    // --- 1. Tín hi?u k?t n?i t? PE (Tr? n?i b?: 3 chu k?) ---
    wire signed [19:0] p0, p1, p2, p3;

    // K?t n?i port theo v? trí chính xác: (clk, x, w, p)
    pe_direct_conv2 pe0 (clk, x0, w0, p0);
    pe_direct_conv2 pe1 (clk, x1, w1, p1);
    pe_direct_conv2 pe2 (clk, x2, w2, p2);
    pe_direct_conv2 pe3 (clk, x3, w3, p3);

    // --- 2. Các thanh ghi trung gian cho cây c?ng Pipeline ---
    reg signed [19:0] sum_01, sum_23; // T?ng 1 cây c?ng
    reg signed [19:0] sum_total;     // T?ng 2 cây c?ng
    reg signed [19:0] bias_d1, bias_d2; // Chu?i thanh ghi d?ch tr? cho Bias
    reg signed [19:0] out_reg;       // T?ng cu?i ch?t ngõ ra

    // --- 3. Logic cây c?ng Pipeline vi?t tr?c ti?p b?ng toán t? + ---
    always @(posedge clk) begin
        // T?NG 1: C?ng c?p p0+p1 và p2+p3 (Tr? 1 clk t? sau PE)
        sum_01  <= p0 + p1;
        sum_23  <= p2 + p3;
        
        // D?ch tr? Bias t?ng 1
        bias_d1 <= bias;

        // T?NG 2: C?ng g?p 2 nhánh thành sum_total (Tr? 2 clk t? sau PE)
        sum_total <= sum_01 + sum_23;
        
        // D?ch tr? Bias t?ng 2 d? kh?p nh?p d? li?u v?i sum_total
        bias_d2   <= bias_d1;

        // T?NG CU?I: C?ng v?i Bias dã d?ng b? tr? và ch?t ngõ ra (Tr? 3 clk t? sau PE)
        out_reg   <= sum_total + bias_d2;
    end

    // Gán liên t?c ra Port ngõ ra d?ng wire
    // T?ng d? tr? toàn c?c t? Input d?u vào -> neuron_out:
    // 3 (PE n?i b?) + 3 (Cây c?ng & Bias) = 6 chu k? clk.
    assign neuron_out = out_reg;

endmodule

module Dense_Layer_TOP (
    input clk, rst_n, load_en,
    input valid_in,
    input signed [11:0] in_f0, in_f1, in_f2, in_f3,
    input [19:0] weight_in_lut,
    
    output reg out_valid,
    output signed [11:0] out0, out1, out2, out3, out4, out5, out6, out7
) ;

    // --- 1. CHU?I SCAN-CHAIN 40 PH?N T? (Li?t kê tu?ng minh) ---
    reg [19:0] regs [0:39];

  always @(posedge clk ) begin
        if (!rst_n) begin
            // Reset toàn b? 40 thanh ghi v? 0
            regs[0]<=0;  regs[1]<=0;  regs[2]<=0;  regs[3]<=0;  regs[4]<=0;  regs[5]<=0;  regs[6]<=0;  regs[7]<=0;
            regs[8]<=0;  regs[9]<=0;  regs[10]<=0; regs[11]<=0; regs[12]<=0; regs[13]<=0; regs[14]<=0; regs[15]<=0;
            regs[16]<=0; regs[17]<=0; regs[18]<=0; regs[19]<=0; regs[20]<=0; regs[21]<=0; regs[22]<=0; regs[23]<=0;
            regs[24]<=0; regs[25]<=0; regs[26]<=0; regs[27]<=0; regs[28]<=0; regs[29]<=0; regs[30]<=0; regs[31]<=0;
            regs[32]<=0; regs[33]<=0; regs[34]<=0; regs[35]<=0; regs[36]<=0; regs[37]<=0; regs[38]<=0; regs[39]<=0;
        end else if (load_en) begin
            // D?ch d? li?u: C?ng vào n?i v?i regs[0]
            regs[0]  <= weight_in_lut;
            regs[1]  <= regs[0];  regs[2]  <= regs[1];  regs[3]  <= regs[2];  regs[4]  <= regs[3];
            regs[5]  <= regs[4];  regs[6]  <= regs[5];  regs[7]  <= regs[6];  regs[8]  <= regs[7];
            regs[9]  <= regs[8];  regs[10] <= regs[9];  regs[11] <= regs[10]; regs[12] <= regs[11];
            regs[13] <= regs[12]; regs[14] <= regs[13]; regs[15] <= regs[14]; regs[16] <= regs[15];
            regs[17] <= regs[16]; regs[18] <= regs[17]; regs[19] <= regs[18]; regs[20] <= regs[19];
            regs[21] <= regs[20]; regs[22] <= regs[21]; regs[23] <= regs[22]; regs[24] <= regs[23];
            regs[25] <= regs[24]; regs[26] <= regs[25]; regs[27] <= regs[26]; regs[28] <= regs[27];
            regs[29] <= regs[28]; regs[30] <= regs[29]; regs[31] <= regs[30]; regs[32] <= regs[31];
            regs[33] <= regs[32]; regs[34] <= regs[33]; regs[35] <= regs[34]; regs[36] <= regs[35];
            regs[37] <= regs[36]; regs[38] <= regs[37]; regs[39] <= regs[38];
        end
    end

    wire signed [19:0] raw [0:7];

    // Neuron 0
    dense_neuron n0 (
        .clk(clk),
        .x0(in_f0), .x1(in_f1), .x2(in_f2), .x3(in_f3),
        .w0(regs[39]), .w1(regs[31]), .w2(regs[23]), .w3(regs[15]),
        .bias(regs[7]),
        .neuron_out(raw[0])
    );

    // Neuron 1
    dense_neuron n1 (
        .clk(clk),
        .x0(in_f0), .x1(in_f1), .x2(in_f2), .x3(in_f3),
        .w0(regs[38]), .w1(regs[30]), .w2(regs[22]), .w3(regs[14]),
        .bias(regs[6]),
        .neuron_out(raw[1])
    );

    // Neuron 2
    dense_neuron n2 (
        .clk(clk),
        .x0(in_f0), .x1(in_f1), .x2(in_f2), .x3(in_f3),
        .w0(regs[37]), .w1(regs[29]), .w2(regs[21]), .w3(regs[13]),
        .bias(regs[5]),
        .neuron_out(raw[2])
    );

    // Neuron 3
    dense_neuron n3 (
        .clk(clk),
        .x0(in_f0), .x1(in_f1), .x2(in_f2), .x3(in_f3),
        .w0(regs[36]), .w1(regs[28]), .w2(regs[20]), .w3(regs[12]),
        .bias(regs[4]),
        .neuron_out(raw[3])
    );

    // Neuron 4
    dense_neuron n4 (
        .clk(clk),
        .x0(in_f0), .x1(in_f1), .x2(in_f2), .x3(in_f3),
        .w0(regs[35]), .w1(regs[27]), .w2(regs[19]), .w3(regs[11]),
        .bias(regs[3]),
        .neuron_out(raw[4])
    );

    // Neuron 5
    dense_neuron n5 (
        .clk(clk),
        .x0(in_f0), .x1(in_f1), .x2(in_f2), .x3(in_f3),
        .w0(regs[34]), .w1(regs[26]), .w2(regs[18]), .w3(regs[10]),
        .bias(regs[2]),
        .neuron_out(raw[5])
    );

    // Neuron 6
    dense_neuron n6 (
        .clk(clk),
        .x0(in_f0), .x1(in_f1), .x2(in_f2), .x3(in_f3),
        .w0(regs[33]), .w1(regs[25]), .w2(regs[17]), .w3(regs[9]),
        .bias(regs[1]),
        .neuron_out(raw[6])
    );

    // Neuron 7
    dense_neuron n7 (
        .clk(clk),
        .x0(in_f0), .x1(in_f1), .x2(in_f2), .x3(in_f3),
        .w0(regs[32]), .w1(regs[24]), .w2(regs[16]), .w3(regs[8]),
        .bias(regs[0]),
        .neuron_out(raw[7])
    );
  
    wire signed [11:0] relu_out [0:7];
  // Kh?i ReLU (Logic t? h?p)
    ReLU_Unit_conv2 relu0 (raw[0], 20'd0, relu_out[0]);
    ReLU_Unit_conv2 relu1 (raw[1], 20'd0, relu_out[1]);
    ReLU_Unit_conv2 relu2 (raw[2], 20'd0, relu_out[2]);
    ReLU_Unit_conv2 relu3 (raw[3], 20'd0, relu_out[3]);
    ReLU_Unit_conv2 relu4 (raw[4], 20'd0, relu_out[4]);
    ReLU_Unit_conv2 relu5 (raw[5], 20'd0, relu_out[5]);
    ReLU_Unit_conv2 relu6 (raw[6], 20'd0, relu_out[6]);
    ReLU_Unit_conv2 relu7 (raw[7], 20'd0, relu_out[7]);

   // --- 3. PIPELINE REGISTERS (ÐÃ C?P NH?T DELAY 11 CHU K?) ---
    // Phân tích d? tr?:
    // T?ng tr? yêu c?u = 11 chu k?.
    // v_pipe[0] nh?n valid_in t?i T=1, v_pipe[10] s? b?t t?i T=11.

    reg [10:0] v_pipe; // Thanh ghi d?ch 11 t?ng (t? index 0 d?n 10)
    reg signed [11:0] out_reg [0:7];

    // Gán ngõ ra t? thanh ghi d?m
    assign out0 = out_reg[0];
    assign out1 = out_reg[1];
    assign out2 = out_reg[2];
    assign out3 = out_reg[3];
    assign out4 = out_reg[4];
    assign out5 = out_reg[5];
    assign out6 = out_reg[6];
    assign out7 = out_reg[7];

  always @(posedge clk ) begin
        if (!rst_n) begin
            // Reset tín hi?u valid và pipe
            v_pipe    <= 11'd0;
            out_valid <= 1'b0;
            
            // Reset các thanh ghi d? li?u
            out_reg[0] <= 12'd0; out_reg[1] <= 12'd0; out_reg[2] <= 12'd0; out_reg[3] <= 12'd0;
            out_reg[4] <= 12'd0; out_reg[5] <= 12'd0; out_reg[6] <= 12'd0; out_reg[7] <= 12'd0;
        end else begin
            // 1. D?ch tín hi?u Valid vào du?ng ?ng 11 t?ng
            // valid_in n?p vào bit th?p nh?t, các bit cu d?y d?n lên bit cao
            v_pipe <= {v_pipe[9:0], valid_in};

            // 2. Ch?t tín hi?u out_valid ? d?u ra cu?i cùng (T?i nh?p th? 11)
            out_valid <= v_pipe[10];

            // 3. Ch?t d? li?u vào out_reg khi d? li?u dã tính xong
            // Ki?m tra v_pipe[9] d? ? su?n clock ti?p theo (nh?p th? 11), 
            // d? li?u du?c n?p vào out_reg cùng lúc out_valid (v_pipe[10]) lên cao.
            if (v_pipe[9]) begin
                out_reg[0] <= relu_out[0];
                out_reg[1] <= relu_out[1];
                out_reg[2] <= relu_out[2];
                out_reg[3] <= relu_out[3];
                out_reg[4] <= relu_out[4];
                out_reg[5] <= relu_out[5];
                out_reg[6] <= relu_out[6];
                out_reg[7] <= relu_out[7];
            end else begin
                
            end
        end
    end

endmodule

module dense_neuron_8in (
    input  wire        clk,
    input  wire signed [11:0] x0, x1, x2, x3, x4, x5, x6, x7,
    input  wire        [19:0] w0, w1, w2, w3, w4, w5, w6, w7,
    input  wire signed [19:0] bias,
    output wire signed [19:0] neuron_out
);

    // --- 1. Tín hi?u k?t n?i t? PE (Tr? n?i b?: 3 chu k?) ---
    wire signed [19:0] p0, p1, p2, p3, p4, p5, p6, p7;

    // K?t n?i port theo v? trí (dúng c?u trúc clk, x, w, p)
    pe_direct_conv2 pe0 (clk, x0, w0, p0);
    pe_direct_conv2 pe1 (clk, x1, w1, p1);
    pe_direct_conv2 pe2 (clk, x2, w2, p2);
    pe_direct_conv2 pe3 (clk, x3, w3, p3);
    pe_direct_conv2 pe4 (clk, x4, w4, p4);
    pe_direct_conv2 pe5 (clk, x5, w5, p5);
    pe_direct_conv2 pe6 (clk, x6, w6, p6);
    pe_direct_conv2 pe7 (clk, x7, w7, p7);

    // --- 2. Các thanh ghi trung gian cho cây c?ng Pipeline ---
    reg signed [19:0] s1_01, s1_23, s1_45, s1_67; // T?ng 1 cây c?ng
    reg signed [19:0] s2_03, s2_47;               // T?ng 2 cây c?ng
    reg signed [19:0] s3_total;                   // T?ng 3 cây c?ng
    reg signed [19:0] b_d1, b_d2, b_d3;           // Chu?i thanh ghi tr? cho Bias
    reg signed [19:0] out_reg;                    // T?ng cu?i ch?t ngõ ra

    // --- 3. Logic cây c?ng Pipeline không dùng Instance b? c?ng r?i ---
    always @(posedge clk) begin
        // T?NG 1: C?ng c?p 8 ngõ ra t? PE (Tr? 1 clk t? PE)
        s1_01   <= p0 + p1;
        s1_23   <= p2 + p3;
        s1_45   <= p4 + p5;
        s1_67   <= p6 + p7;
        
        // D?ch tr? Bias t?ng 1
        b_d1    <= bias;

        // T?NG 2: C?ng g?p 4 nhánh thành 2 nhánh (Tr? 2 clk t? PE)
        s2_03   <= s1_01 + s1_23;
        s2_47   <= s1_45 + s1_67;
        
        // D?ch tr? Bias t?ng 2
        b_d2    <= b_d1;

        // T?NG 3: C?ng t?ng các tích (Tr? 3 clk t? PE)
        s3_total <= s2_03 + s2_47;
        
        // D?ch tr? Bias t?ng 3 d? kh?p nh?p v?i s3_total
        b_d3     <= b_d2;

        // T?NG CU?I: C?ng v?i Bias dã du?c d?ng b? tr? (Tr? 4 clk t? PE)
        out_reg  <= s3_total + b_d3;
    end

    // Gán liên t?c ra Port ngõ ra d?ng wire
    // T?ng d? tr? t? Input -> neuron_out: 3 (PE) + 4 (Cây c?ng & Bias) = 7 chu k? clk.
    assign neuron_out = out_reg;

endmodule

// --- Kh?i thu nh? và bão hòa d? li?u sang 12-bit signed cho l?p tuy?n tính ---
module Linear_Scale_Unit (
    input  wire signed [19:0] data_in,   // K?t qu? t? Neuron (dã có bias)
    output wire signed [11:0] data_out   // Ngõ ra 12-bit s? có d?u
);

    // Thay th? b? c?ng r?i b?ng toán t? c?ng tr?c ti?p. 
    // Trình t?ng h?p s? t? d?ng map vào m?ch Carry Chain hi?u nang cao.
    wire signed [19:0] sum_round = data_in + 20'sd16;

    // D?ch ph?i s? h?c b?o toàn bit d?u (Shift = 5)
    wire signed [19:0] s_shifted = sum_round >>> 5;

    // Saturation (Bão hòa s? có d?u 12-bit: -2048 d?n 2047)
    // S?a l?i cú pháp h?ng s? âm ch?n du?i d?m b?o chính xác (-12'sd2048)
    assign data_out = (s_shifted > 20'sd2047)  ? 12'sd2047 : 
                      (s_shifted < -20'sd2048) ? -12'sd2048 : 
                      s_shifted[11:0];

endmodule

module Dense_Layer_2_TOP (
    input clk, rst_n, load_en,
    input valid_in,
    input signed [11:0] in0, in1, in2, in3, in4, in5, in6, in7,
    input [19:0] weight_in_lut,
    
    output reg out_valid,
    output signed [11:0] out0, out1, out2, out3, out4, out5
);

    // --- 1. CHU?I SCAN-CHAIN 54 PH?N T? ---
    reg [19:0] r [0:53];

  always @(posedge clk ) begin
        if (!rst_n) begin
            r[0]<=0; r[1]<=0; r[2]<=0; r[3]<=0; r[4]<=0; r[5]<=0; r[6]<=0; r[7]<=0; r[8]<=0; r[9]<=0;
            r[10]<=0;r[11]<=0;r[12]<=0;r[13]<=0;r[14]<=0;r[15]<=0;r[16]<=0;r[17]<=0;r[18]<=0;r[19]<=0;
            r[20]<=0;r[21]<=0;r[22]<=0;r[23]<=0;r[24]<=0;r[25]<=0;r[26]<=0;r[27]<=0;r[28]<=0;r[29]<=0;
            r[30]<=0;r[31]<=0;r[32]<=0;r[33]<=0;r[34]<=0;r[35]<=0;r[36]<=0;r[37]<=0;r[38]<=0;r[39]<=0;
            r[40]<=0;r[41]<=0;r[42]<=0;r[43]<=0;r[44]<=0;r[45]<=0;r[46]<=0;r[47]<=0;r[48]<=0;r[49]<=0;
            r[50]<=0;r[51]<=0;r[52]<=0;r[53]<=0;
        end else if (load_en) begin
            r[0] <= weight_in_lut;
            r[1]<=r[0];  r[2]<=r[1];  r[3]<=r[2];  r[4]<=r[3];  r[5]<=r[4];  r[6]<=r[5];  r[7]<=r[6];  r[8]<=r[7];  r[9]<=r[8];
            r[10]<=r[9]; r[11]<=r[10];r[12]<=r[11];r[13]<=r[12];r[14]<=r[13];r[15]<=r[14];r[16]<=r[15];r[17]<=r[16];r[18]<=r[17];r[19]<=r[18];
            r[20]<=r[19];r[21]<=r[20];r[22]<=r[21];r[23]<=r[22];r[24]<=r[23];r[25]<=r[24];r[26]<=r[25];r[27]<=r[26];r[28]<=r[27];r[29]<=r[28];
            r[30]<=r[29];r[31]<=r[30];r[32]<=r[31];r[33]<=r[32];r[34]<=r[33];r[35]<=r[34];r[36]<=r[35];r[37]<=r[36];r[38]<=r[37];r[39]<=r[38];
            r[40]<=r[39];r[41]<=r[40];r[42]<=r[41];r[43]<=r[42];r[44]<=r[43];r[45]<=r[44];r[46]<=r[45];r[47]<=r[46];r[48]<=r[47];r[49]<=r[48];
            r[50]<=r[49];r[51]<=r[50];r[52]<=r[51];r[53]<=r[52];
        end
    end

    // --- 2. MAPPING NEURON (8 INPUT, 6 OUTPUT -> STEP 6) ---
    // --- 2. MAPPING NEURON (8 INPUT, 6 OUTPUT -> STEP 6) ---
    // S? d?ng b? c?ng Pipeline: Ð? tr? x? lý là 8 chu k? clock (4 t?ng adder)
    wire signed [19:0] raw [0:5];

    // Neuron 0
    dense_neuron_8in n0 (
        .clk(clk),
        .x0(in0), .x1(in1), .x2(in2), .x3(in3), .x4(in4), .x5(in5), .x6(in6), .x7(in7),
        .w0(r[53]), .w1(r[47]), .w2(r[41]), .w3(r[35]), .w4(r[29]), .w5(r[23]), .w6(r[17]), .w7(r[11]),
        .bias(r[5]), 
        .neuron_out(raw[0])
    );

    // Neuron 1
    dense_neuron_8in n1 (
        .clk(clk),
        .x0(in0), .x1(in1), .x2(in2), .x3(in3), .x4(in4), .x5(in5), .x6(in6), .x7(in7),
        .w0(r[52]), .w1(r[46]), .w2(r[40]), .w3(r[34]), .w4(r[28]), .w5(r[22]), .w6(r[16]), .w7(r[10]),
        .bias(r[4]), 
        .neuron_out(raw[1])
    );

    // Neuron 2
    dense_neuron_8in n2 (
        .clk(clk),
        .x0(in0), .x1(in1), .x2(in2), .x3(in3), .x4(in4), .x5(in5), .x6(in6), .x7(in7),
        .w0(r[51]), .w1(r[45]), .w2(r[39]), .w3(r[33]), .w4(r[27]), .w5(r[21]), .w6(r[15]), .w7(r[9]),
        .bias(r[3]), 
        .neuron_out(raw[2])
    );

    // Neuron 3
    dense_neuron_8in n3 (
        .clk(clk),
        .x0(in0), .x1(in1), .x2(in2), .x3(in3), .x4(in4), .x5(in5), .x6(in6), .x7(in7),
        .w0(r[50]), .w1(r[44]), .w2(r[38]), .w3(r[32]), .w4(r[26]), .w5(r[20]), .w6(r[14]), .w7(r[8]),
        .bias(r[2]), 
        .neuron_out(raw[3])
    );

    // Neuron 4
    dense_neuron_8in n4 (
        .clk(clk),
        .x0(in0), .x1(in1), .x2(in2), .x3(in3), .x4(in4), .x5(in5), .x6(in6), .x7(in7),
        .w0(r[49]), .w1(r[43]), .w2(r[37]), .w3(r[31]), .w4(r[25]), .w5(r[19]), .w6(r[13]), .w7(r[7]),
        .bias(r[1]), 
        .neuron_out(raw[4])
    );

    // Neuron 5
    dense_neuron_8in n5 (
        .clk(clk),
        .x0(in0), .x1(in1), .x2(in2), .x3(in3), .x4(in4), .x5(in5), .x6(in6), .x7(in7),
        .w0(r[48]), .w1(r[42]), .w2(r[36]), .w3(r[30]), .w4(r[24]), .w5(r[18]), .w6(r[12]), .w7(r[6]),
        .bias(r[0]), 
        .neuron_out(raw[5])
    );

  	wire signed [11:0] scale_out [0:5];
    //Kh?i Scale & Saturation (Logic t? h?p)
    Linear_Scale_Unit scale0 (raw[0], scale_out[0]);
    Linear_Scale_Unit scale1 (raw[1], scale_out[1]);
    Linear_Scale_Unit scale2 (raw[2], scale_out[2]);
    Linear_Scale_Unit scale3 (raw[3], scale_out[3]);
    Linear_Scale_Unit scale4 (raw[4], scale_out[4]);
    Linear_Scale_Unit scale5 (raw[5], scale_out[5]);

    reg [10:0] v_pipe_d2; // Thanh ghi d?ch 11 t?ng cho kh?i Dense 2
    reg signed [11:0] out_reg [0:5];

    // Gán ngõ ra t? thanh ghi d?m ra Port c?a Module
    assign out0 = out_reg[0];
    assign out1 = out_reg[1];
    assign out2 = out_reg[2];
    assign out3 = out_reg[3];
    assign out4 = out_reg[4];
    assign out5 = out_reg[5];

  always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v_pipe_d2 <= 11'd0;
            out_valid <= 1'b0;
            out_reg[0] <= 12'd0; out_reg[1] <= 12'd0; out_reg[2] <= 12'd0;
            out_reg[3] <= 12'd0; out_reg[4] <= 12'd0; out_reg[5] <= 12'd0;
        end else begin
            // 1. D?ch tín hi?u Valid vào du?ng ?ng 11 t?ng
            v_pipe_d2 <= {v_pipe_d2[9:0], valid_in};
            out_valid <= v_pipe_d2[10];

            if (v_pipe_d2[9]) begin
                out_reg[0] <= scale_out[0];
                out_reg[1] <= scale_out[1];
                out_reg[2] <= scale_out[2];
                out_reg[3] <= scale_out[3];
                out_reg[4] <= scale_out[4];
                out_reg[5] <= scale_out[5];
            end else begin
            end
        end
    end

endmodule

module Argmax_6 (
    input clk, rst_n,
    input valid_in,
    input signed [11:0] in0, in1, in2, in3, in4, in5,
    output reg [2:0] class_out,
    output reg valid_out
);

    // =========================
    // Stage 1
    // =========================
    reg signed [11:0] val_a, val_b, val_c;
    reg [5:0] idx_a_oh, idx_b_oh, idx_c_oh;
    reg v1_reg;

  always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            val_a <= 12'sd0; val_b <= 12'sd0; val_c <= 12'sd0;
            idx_a_oh <= 6'b0;
            idx_b_oh <= 6'b0;
            idx_c_oh <= 6'b0;
            v1_reg <= 1'b0;
        end else begin
            v1_reg <= valid_in;

            if (valid_in) begin
                // Pair 0-1
                if (in0 >= in1) begin
                    val_a <= in0;
                    idx_a_oh <= 6'b000001;
                end else begin
                    val_a <= in1;
                    idx_a_oh <= 6'b000010;
                end

                // Pair 2-3
                if (in2 >= in3) begin
                    val_b <= in2;
                    idx_b_oh <= 6'b000100;
                end else begin
                    val_b <= in3;
                    idx_b_oh <= 6'b001000;
                end

                // Pair 4-5
                if (in4 >= in5) begin
                    val_c <= in4;
                    idx_c_oh <= 6'b010000;
                end else begin
                    val_c <= in5;
                    idx_c_oh <= 6'b100000;
                end
            end
        end
    end

    // =========================
    // Stage 2
    // =========================
    reg signed [11:0] val_max;
    reg [5:0] idx_max_oh;
    reg v2_reg;

  always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            val_max <= 12'sd0;
            idx_max_oh <= 6'b0;
            v2_reg <= 1'b0;
        end else begin
            v2_reg <= v1_reg;

            if (v1_reg) begin
                if (val_a >= val_b && val_a >= val_c) begin
                    val_max <= val_a;
                    idx_max_oh <= idx_a_oh;
                end else if (val_b >= val_a && val_b >= val_c) begin
                    val_max <= val_b;
                    idx_max_oh <= idx_b_oh;
                end else begin
                    val_max <= val_c;
                    idx_max_oh <= idx_c_oh;
                end
            end
        end
    end

    // =========================
    // Stage 3: one-hot ? binary (NO function)
    // =========================
    always @(posedge clk) begin
        if (!rst_n) begin
            class_out <= 3'd0;
            valid_out <= 1'b0;
        end else begin
            valid_out <= v2_reg;

            if (v2_reg) begin
                case (idx_max_oh)
                    6'b000001: class_out <= 3'd0;
                    6'b000010: class_out <= 3'd1;
                    6'b000100: class_out <= 3'd2;
                    6'b001000: class_out <= 3'd3;
                    6'b010000: class_out <= 3'd4;
                    6'b100000: class_out <= 3'd5;
                    default:   class_out <= 3'd0;
                endcase
            end
        end
    end

endmodule

/*********************************************************************
* Module: PingPong_Wrapper
* Ch?c nang: Qu?n lý 2 SRAM 128x48, th?c hi?n Tách/Ghép d? li?u 48-bit
* và hoán d?i vai trò Read/Write gi?a các Layer.
**********************************************************************/
module SRAM_24to48_Bridge (
    input [6:0] A,
    input CLK,
    input WE,
    input [47:0] D,
    output [47:0] Q
);
    // Ghép 2 con SRAM 24-bit d? t?o thành 48-bit
    // SRAM Low: bit [23:0], SRAM High: bit [47:24]
    
    RSPB18_128X24M4_G1 mem_low (
        .Q(Q[23:0]),
        .ADR(A),
        .D(D[23:0]),
        .WE(WE),      // 1: Write, 0: Read
        .WEM(3'b111), // Ghi toàn b? 24 bit
        .OE(1'b1),    // Luôn m? Output
        .ME(1'b1),    // Luôn ch?n Chip
        .CLK(CLK)
    );

    RSPB18_128X24M4_G1 mem_high (
        .Q(Q[47:24]),
        .ADR(A),
        .D(D[47:24]),
        .WE(WE),
        .WEM(3'b111),
        .OE(1'b1),
        .ME(1'b1),
        .CLK(CLK)
    );
endmodule

module PingPong_Wrapper (
    input  clk,
    input  rst_n,           // S? d?ng tín hi?u reset h? th?ng
    input  select,           
    input  [6:0] rd_addr,    
    input  [6:0] wr_addr,    
    input  wr_en,            
    input  [11:0] in_f0, in_f1, in_f2, in_f3,
    output [11:0] out_f0, out_f1, out_f2, out_f3
);

    wire [47:0] din_combined = {in_f3, in_f2, in_f1, in_f0};
    wire [47:0] dout_a, dout_b;

    // --- Logic Reset cho các tín hi?u di?u khi?n ---
    // N?u rst_n = 0, ép các tín hi?u v? giá tr? an toàn (0) 
    // d? SRAM không báo l?i "WE unknown" ho?c "ADR unknown"
    
    wire [6:0] addr_a_int = (select == 1'b0) ? rd_addr : wr_addr;
    wire [6:0] addr_a     = (rst_n) ? addr_a_int : 7'd0;
    
    wire we_a_int         = (select == 1'b1) ? wr_en : 1'b0;
    wire we_a             = (rst_n) ? we_a_int : 1'b0;

    wire [6:0] addr_b_int = (select == 1'b1) ? rd_addr : wr_addr;
    wire [6:0] addr_b     = (rst_n) ? addr_b_int : 7'd0;
    
    wire we_b_int         = (select == 1'b0) ? wr_en : 1'b0;
    wire we_b             = (rst_n) ? we_b_int : 1'b0;

    // S? d?ng Bridge 48-bit (ghép t? 2 con 24-bit)
    SRAM_24to48_Bridge mem_group_A (
        .A(addr_a),
        .CLK(clk),
        .WE(we_a),
        .D(din_combined),
        .Q(dout_a)
    );

    SRAM_24to48_Bridge mem_group_B (
        .A(addr_b),
        .CLK(clk),
        .WE(we_b),
        .D(din_combined),
        .Q(dout_b)
    );

    // Xu?t d? li?u (Output logic cung c?n b?o v? rst_n n?u c?n)
    assign {out_f3, out_f2, out_f1, out_f0} = (!rst_n) ? 48'd0 : 
                                              (select == 1'b0) ? dout_a : dout_b;

endmodule

module Zero_Gating (
    input  wire        clk,
    input  wire [6:0]  rd_addr,     // Ð?a ch? t? Controller
    input  wire [11:0] in_f0, in_f1, in_f2, in_f3, // D? li?u t? RAM
    output wire [11:0] out_f0, out_f1, out_f2, out_f3 // D? li?u dã l?c
);
    reg [6:0] rd_addr_reg;
    always @(posedge clk) rd_addr_reg <= rd_addr;

    // N?u d?a ch? là 127 (d?a ch? ?o), ép d? li?u v? 0 d? tránh nhi?u/sai s?
    assign out_f0 = (rd_addr_reg == 7'd127) ? 12'd0 : in_f0;
    assign out_f1 = (rd_addr_reg == 7'd127) ? 12'd0 : in_f1;
    assign out_f2 = (rd_addr_reg == 7'd127) ? 12'd0 : in_f2;
    assign out_f3 = (rd_addr_reg == 7'd127) ? 12'd0 : in_f3;
endmodule

module Conv_Output_Mux (
    input  wire        is_layer0_sel,
    input  wire [11:0] l1_f0, l1_f1, l1_f2, l1_f3,
    input  wire [11:0] lr_f0, lr_f1, lr_f2, lr_f3,
    output wire [11:0] out_f0, out_f1, out_f2, out_f3
);
    assign out_f0 = is_layer0_sel ? l1_f0 : lr_f0;
    assign out_f1 = is_layer0_sel ? l1_f1 : lr_f1;
    assign out_f2 = is_layer0_sel ? l1_f2 : lr_f2;
    assign out_f3 = is_layer0_sel ? l1_f3 : lr_f3;
endmodule

module Neural_Weight_Manager (
    input  wire        clk,
    input  wire        rst_n,
    
    // Giao di?n di?u khi?n (T? CPU ho?c Testbench)
    input  wire        config_mode, 
    input  wire [8:0]  cpu_addr,    
    input  wire [19:0] cpu_data_in, 
    input  wire        cpu_wr_en,   
    
    // Giao di?n t? CNN Controller (Ch? d?c)
    input  wire [8:0]  cnn_addr,
    output wire [19:0] csd_data_out
);
    wire sram_me = rst_n; 
    wire [8:0]  sram_addr_sel = config_mode ? cpu_addr  : cnn_addr;
    wire        sram_we_sel   = config_mode ? cpu_wr_en : 1'b0;
    
    wire [8:0]  sram_addr     = (rst_n) ? sram_addr_sel : 9'd0;
    wire        sram_we       = (rst_n) ? sram_we_sel   : 1'b0;
    
  wire [23:0] sram_din      = (rst_n) ? {4'd0, cpu_data_in} : 24'd0;
  wire [23:0] sram_dout;
    RSPB18_512X24M4_G1 u_weight_sram (
        .CLK  (clk),
        .ME   (sram_me),     // S? d?ng tín hi?u ME dã qua x? lý rst_n
        .WE   (sram_we),     
        .ADR  (sram_addr),   
        .D    (sram_din),    
      .WEM  (3'b111),     // C?p nh?t m?t n? ghi thành 4-bit d? cho phép ghi d? 32-bit
        .OE   (1'b1),        
        .Q    (sram_dout)    
    );

    assign csd_data_out = sram_dout[19:0];

endmodule

`default_nettype none

module RSPB18_512X24M4_G1(
    input  wire        CLK,
    input  wire        ME,
    input  wire [8:0]  ADR,
    input  wire        WE,
  input  wire [2:0]  WEM,
  input  wire [23:0] D,
    input  wire        OE,
  output wire [23:0] Q
);
    generate
        genvar i;
      for (i = 0; i < 3; i = i + 1) begin : gen_bram
            wire [7:0] q_byte;  // wire trung gian cho t?ng byte
            singleport_bram #(.DEPTH(512)) bram_inst (
                .CLK (CLK),
                .ME  (ME),
                .ADR (ADR),
                .WE  (WE & WEM[i]),
                .D   (D[8*i +: 8]),
                .Q   (q_byte),
                .OE  (OE)
            );
            assign Q[8*i +: 8] = q_byte;
        end
    endgenerate
endmodule

module RSPB18_128X24M4_G1(
    input  wire        CLK,
    input  wire        ME,
    input  wire [6:0]  ADR,
    input  wire        WE,
  input  wire [2:0]  WEM,
  input  wire [23:0] D,
    input  wire        OE,
  output wire [23:0] Q
);
    generate
        genvar i;
      for (i = 0; i < 3; i = i + 1) begin : gen_bram
            wire [7:0] q_byte;
            singleport_bram #(.DEPTH(128)) bram_inst (
                .CLK (CLK),
                .ME  (ME),
                .ADR (ADR),
                .WE  (WE & WEM[i]),
                .D   (D[8*i +: 8]),
                .Q   (q_byte),
                .OE  (OE)
            );
            assign Q[8*i +: 8] = q_byte;
        end
    endgenerate
endmodule

module singleport_bram #(
    parameter DEPTH = 512
)(
    input wire CLK,
    input wire ME,                      // Master Enable
    input wire [$clog2(DEPTH)-1:0] ADR, // Address
    input wire WE,                      // Write Enable
    input wire [7:0] D,                 // Write Data
    input wire OE,                      // Output Enable
    output reg [7:0] Q                  // Read Data
);
    reg [7:0] mem [0:DEPTH-1]; 

    always @(posedge CLK) begin
        if (ME) begin
            if (WE) begin
                mem[ADR] <= D;
            end else begin
                Q <= mem[ADR];
            end
        end
    end
endmodule


module CNN_Global_Controller (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    
    output reg  [8:0]  lut_addr,
    output reg  [6:0]  rd_addr,
    output reg  [6:0]  wr_addr,
    output reg         select_pp,
    
    output reg  [2:0]  curr_layer,
    output reg         ready_out,
    output reg         done_system,
    
    input  wire        l1_v_pool,
    input  wire        lr_v_pool,
    input  wire        gap_valid,
    input  wire        d1_valid,
    input  wire        d2_valid,
    input  wire        done_argmax,
    
    output wire        l0_load_en,
    output wire        lr_load_en,
    output wire        lr_valid_to_conv,
    output wire        is_layer0_sel,
    output wire        pp_wr_en,
    output wire        gap_valid_in,
    output wire        d1_load_en,
    output wire        d1_valid_in,
    output wire        d2_load_en,
    output wire        d2_valid_in
);

    localparam IDLE     = 3'd0,
               LOAD     = 3'd1,
               DATA     = 3'd2,
               DRAIN    = 3'd3,
               DONE     = 3'd4,
               WAIT_GAP = 3'd5;

    reg [2:0] state;
    reg [7:0] w_cnt;
    reg [7:0] d_cnt;
    reg [6:0] drain_cnt;
    reg       valid_in_node;
    reg       layer_ready;
    reg       load_en_node;
    reg       gap_locked;
    reg       load_en_node_delay;
    reg [3:0] gap_sample_cnt;

    wire is_l0     = (curr_layer == 3'd0);
    wire is_reuse  = (curr_layer >= 3'd1 && curr_layer <= 3'd3);
    wire is_dense1 = (curr_layer == 3'd4);
    wire is_dense2 = (curr_layer == 3'd5);

    wire internal_v_pool = is_l0 ? l1_v_pool : (is_reuse ? lr_v_pool : 1'b0);

    wire ctrl_feedback_v = (curr_layer < 3'd3)  ? internal_v_pool :
                           (curr_layer == 3'd3) ? gap_valid        :
                           (is_dense1)          ? d1_valid         : d2_valid;

    wire [7:0] w_max = is_l0     ? 8'd27  :
                       is_reuse  ? 8'd111 :
                       is_dense1 ? 8'd39  : 8'd53;

    wire [8:0] layer_offset = (curr_layer == 3'd0) ? 9'd0   :
                              (curr_layer == 3'd1) ? 9'd28  :
                              (curr_layer == 3'd2) ? 9'd140 :
                              (curr_layer == 3'd3) ? 9'd252 :
                              (curr_layer == 3'd4) ? 9'd364 : 9'd404;

    wire [7:0] d_max = (curr_layer == 3'd0) ? 8'd205 :
                       (curr_layer == 3'd1) ? 8'd90  :
                       (curr_layer == 3'd2) ? 8'd36  : 8'd32;

    wire [6:0] max_rd_addr = (curr_layer == 3'd1) ? 7'd65 :
                             (curr_layer == 3'd2) ? 7'd21 : 7'd6;

    always @(posedge clk ) begin
        if (!rst_n) begin
            state              <= IDLE;
            curr_layer         <= 3'd0;
            select_pp          <= 1'b0;
            done_system        <= 1'b0;
            load_en_node       <= 1'b0;
            load_en_node_delay <= 1'b0;
            valid_in_node      <= 1'b0;
            rd_addr            <= 7'd127;
            wr_addr            <= 7'd0;
            w_cnt              <= 8'd0;
            d_cnt              <= 8'd0;
            drain_cnt          <= 7'd0;
            lut_addr           <= 9'd0;
            layer_ready        <= 1'b0;
            gap_locked         <= 1'b0;
            gap_sample_cnt     <= 4'd0;
            ready_out          <= 1'b0;
        end else begin
            load_en_node_delay <= load_en_node;
            layer_ready        <= 1'b0;

            // gap_sample_cnt reset
            if (state == IDLE || state == DONE)
                gap_sample_cnt <= 4'd0;
            else if (curr_layer == 3'd3 && internal_v_pool)
                gap_sample_cnt <= gap_sample_cnt + 4'd1;

            // gap_locked
            if (curr_layer == 3'd3 && gap_valid)
                gap_locked <= 1'b1;

            // wr_addr
            if (state == LOAD || state == IDLE)
                wr_addr <= 7'd0;
            else if (internal_v_pool && curr_layer < 3'd3)
                wr_addr <= wr_addr + 7'd1;

            case (state)
                IDLE: begin
                    done_system <= 1'b0;
                    w_cnt       <= 8'd0;
                    d_cnt       <= 8'd0;
                    drain_cnt   <= 7'd0;
                    rd_addr     <= 7'd127;
                    gap_locked  <= 1'b0;
                    ready_out   <= 1'b0;
                    if (start) begin
                        state    <= LOAD;
                        lut_addr <= layer_offset;
                    end
                end

                LOAD: begin
                    if (w_cnt == w_max + 8'd1)
                        load_en_node <= 1'b0;
                    else if (w_cnt < w_max + 8'd1)
                        load_en_node <= 1'b1;

                    if (w_cnt == (w_max + 8'd2)) begin
                        w_cnt         <= 8'd0;
                        valid_in_node <= 1'b0;
                        if (curr_layer >= 3'd4) begin
                            state       <= WAIT_GAP;
                            layer_ready <= 1'b1;
                            ready_out   <= 1'b0;
                        end else begin
                            state   <= DATA;
                            rd_addr <= 7'd127;
                            ready_out <= is_l0 ? 1'b1 : 1'b0;
                        end
                    end else begin
                        ready_out <= 1'b0;
                        w_cnt <= w_cnt + 8'd1;
                        if (w_cnt < w_max + 8'd2)
                            lut_addr <= layer_offset + {{1{1'b0}}, w_cnt};
                    end
                end

                DATA: begin
                    ready_out     <= is_l0 ? 1'b1 : 1'b0;
                    valid_in_node <= 1'b1;

                    if (is_l0)
                        rd_addr <= 7'd0;
                    else begin
                        if (d_cnt <= 8'd1)
                            rd_addr <= 7'd127;
                        else if (d_cnt == 8'd2)
                            rd_addr <= 7'd0;
                        else if ((d_cnt - 8'd2) <= {1'b0, max_rd_addr})
                            rd_addr <= d_cnt[6:0] - 7'd2;
                        else
                            rd_addr <= 7'd127;
                    end

                    if (d_cnt == (d_max + 8'd1)) begin
                        d_cnt         <= 8'd0;
                        valid_in_node <= 1'b0;
                        ready_out     <= 1'b0;
                        state         <= DRAIN;
                    end else
                        d_cnt <= d_cnt + 8'd1;
                end

                DRAIN: begin
                    rd_addr   <= 7'd127;
                    ready_out <= 1'b0;
                    if (drain_cnt == 7'd15) begin
                        drain_cnt <= 7'd0;
                        state     <= DONE;
                    end else
                        drain_cnt <= drain_cnt + 7'd1;
                end

                WAIT_GAP: begin
                    ready_out <= 1'b0;
                    if (ctrl_feedback_v) begin
                        if (is_dense1)
                            state <= DONE;
                        else if (is_dense2 && done_argmax) begin
                            done_system <= 1'b1;
                            state       <= IDLE;
                        end
                    end
                end

                DONE: begin
                    d_cnt         <= 8'd0;
                    valid_in_node <= 1'b0;
                    ready_out     <= 1'b0;
                    if (is_dense2)
                        state <= IDLE;
                    else begin
                        if (curr_layer < 3'd3)
                            select_pp <= ~select_pp;
                        curr_layer <= curr_layer + 3'd1;
                        state      <= LOAD;
                        w_cnt      <= 8'd0;
                        case (curr_layer)
                            3'd0: lut_addr <= 9'd28;
                            3'd1: lut_addr <= 9'd140;
                            3'd2: lut_addr <= 9'd252;
                            3'd3: lut_addr <= 9'd364;
                            3'd4: lut_addr <= 9'd404;
                            default: lut_addr <= 9'd0;
                        endcase
                    end
                end

                default: state <= IDLE;
            endcase
        end
    end

    assign is_layer0_sel    = is_l0;
    assign l0_load_en       = load_en_node_delay && is_l0;
    assign lr_load_en       = load_en_node_delay && is_reuse;
    assign d1_load_en       = load_en_node_delay && is_dense1;
    assign d2_load_en       = load_en_node_delay && is_dense2;
    assign lr_valid_to_conv = valid_in_node && is_reuse;
    assign pp_wr_en         = internal_v_pool && (curr_layer < 3'd3);

    assign gap_valid_in = internal_v_pool        &&
                          (curr_layer == 3'd3)   &&
                          (gap_sample_cnt == 4'd2 || gap_sample_cnt == 4'd3) &&
                          !load_en_node_delay    &&
                          !gap_locked;

    assign d1_valid_in = layer_ready && is_dense1;
    assign d2_valid_in = layer_ready && is_dense2;

endmodule
`default_nettype wire



