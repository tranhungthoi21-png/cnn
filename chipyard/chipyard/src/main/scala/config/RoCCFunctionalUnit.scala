//package tut_5
package chipyard

import chisel3._
import chisel3.util.HasBlackBoxResource
import freechips.rocketchip.tile.{BuildRoCC, LazyRoCC, LazyRoCCModuleImp, OpcodeSet, RoCCIO}
import freechips.rocketchip.rocket._
import freechips.rocketchip.diplomacy.LazyModule
import org.chipsalliance.cde.config.{Config, Parameters}
import chisel3.experimental.IntParam
//case object myroccXLen extends Field[Int]

class mymemIO (xLen: Int = 64)(implicit p: Parameters) extends Bundle {
  val clk           = Input(Clock())
  val reset         = Input(UInt(1.W))
  val interrupt     = Output(Bool())
  val busy          = Output(Bool())
  // Command interface
  val cmd_ready     = Output(Bool())
  val cmd_valid     = Input(Bool())
  val cmd_rs1       = Input(UInt(xLen.W))
  val cmd_rs2       = Input(UInt(xLen.W))
  val instr_funct   = Input(UInt(7.W))
  val instr_rs2     = Input(UInt(5.W))
  val instr_rs1     = Input(UInt(5.W))
  val instr_xd      = Input(Bool())
  val instr_xs1     = Input(Bool())
  val instr_xs2     = Input(Bool())
  val instr_rd      = Input(UInt(5.W))
  val instr_opcode  = Input(UInt(7.W))
  // Response interface
  val resp_valid    = Output(Bool())
  val resp_rd       = Output(UInt(5.W))
  val resp_data     = Output(UInt(xLen.W))
}

class RoCCBlackBoxIO (xLen: Int = 64)(implicit p: Parameters) extends Bundle{
  // Control signals
  val clock = Input(Clock())
  val reset = Input(Bool())

  // RoCC interface
  val rocc = new RoCCIO(0, 0)

  // Decoupler-Controller interface
  val bb = new mymemIO(xLen)
}

class platform_bb (xLen: Int = 64)(implicit p: Parameters) extends BlackBox(Map("XLEN" -> IntParam(xLen))) with HasBlackBoxResource {
  val io = IO(new mymemIO(xLen))

  addResource("/vsrc/platform_bb.v")
}

class MyRoccAccel (opcodes: OpcodeSet)(implicit p: Parameters) extends LazyRoCC (
  opcodes   = opcodes,
  nPTWPorts = 0,
  usesFPU   = false,
  roccCSRs  = Nil) {
  override lazy val module = new MyRoccAccelImp(this)
}

class MyRoccAccelImp(outer: MyRoccAccel)(implicit p: Parameters) extends LazyRoCCModuleImp(outer) {
  val xLen = 64

  // Instantiate the rocc modules
  val inst   = Module(new platform_bb(xLen))

  inst.io.clk   := clock
  inst.io.reset := reset.asBool

  // Process cmd
  io.cmd.ready   := inst.io.cmd_ready

  inst.io.cmd_valid     := io.cmd.valid
  inst.io.cmd_rs1       := io.cmd.bits.rs1
  inst.io.cmd_rs2       := io.cmd.bits.rs2
  inst.io.instr_funct   := io.cmd.bits.inst.funct
  inst.io.instr_rs2     := io.cmd.bits.inst.rs2
  inst.io.instr_rs1     := io.cmd.bits.inst.rs1
  inst.io.instr_xd      := io.cmd.bits.inst.xd
  inst.io.instr_xs1     := io.cmd.bits.inst.xs1
  inst.io.instr_xs2     := io.cmd.bits.inst.xs2
  inst.io.instr_rd      := io.cmd.bits.inst.rd
  inst.io.instr_opcode  := io.cmd.bits.inst.opcode
  
  // Process response
  io.resp.valid     := inst.io.resp_valid
  io.resp.bits.rd   := inst.io.resp_rd
  io.resp.bits.data := inst.io.resp_data
  
  io.interrupt      := inst.io.busy
  io.busy           := inst.io.interrupt
  
  // Cache request
  io.mem.req.valid          := false.B
  io.mem.req.bits.addr      := 0.U
  io.mem.req.bits.tag       := 0.U
  io.mem.req.bits.cmd       := M_XRD
  io.mem.req.bits.size      := "b00".U
  io.mem.req.bits.signed    := false.B
  io.mem.req.bits.dprv      := "b00".U
  io.mem.req.bits.dv        := false.B
  io.mem.req.bits.data      := 0.U
  io.mem.req.bits.mask      := 0.U
  io.mem.req.bits.phys      := false.B
  io.mem.req.bits.no_alloc  := false.B
  io.mem.req.bits.no_xcpt   := false.B
  io.mem.s1_kill            := false.B
  io.mem.s1_data.data       := 0.U
  io.mem.s1_data.mask       := 0.U
  io.mem.s2_kill            := false.B
  io.mem.keep_clock_enabled := false.B

  // FPU request
  io.fpu_req.valid            := false.B
  io.fpu_req.bits.rm          := 0.U
  io.fpu_req.bits.fmaCmd      := 0.U
  io.fpu_req.bits.typ         := 0.U
  io.fpu_req.bits.fmt         := 0.U
  io.fpu_req.bits.in1         := 0.U
  io.fpu_req.bits.in2         := 0.U
  io.fpu_req.bits.in3         := 0.U
  io.fpu_req.bits.ldst        := false.B
  io.fpu_req.bits.wen         := false.B
  io.fpu_req.bits.ren1        := false.B
  io.fpu_req.bits.ren2        := false.B
  io.fpu_req.bits.ren3        := false.B
  io.fpu_req.bits.swap12      := false.B
  io.fpu_req.bits.swap23      := false.B
  io.fpu_req.bits.typeTagIn   := 0.U
  io.fpu_req.bits.typeTagOut  := 0.U
  io.fpu_req.bits.fromint     := false.B
  io.fpu_req.bits.toint       := false.B
  io.fpu_req.bits.fastpipe    := false.B
  io.fpu_req.bits.fma         := false.B
  io.fpu_req.bits.div         := false.B
  io.fpu_req.bits.sqrt        := false.B
  io.fpu_req.bits.wflags      := false.B
  io.fpu_resp.ready           := false.B
}

class WithTut05RoccAccel extends Config ((site, here, up) => {
  case BuildRoCC => up(BuildRoCC) ++ Seq(
    (p: Parameters) => {
      val blackbox = LazyModule.apply(new MyRoccAccel(OpcodeSet.all)(p))
      blackbox
    }
  )
})