package chipyard.CNNmod

import Chisel._
import chisel3.WireInit
import chisel3.util.HasBlackBoxResource
import freechips.rocketchip.diplomacy._
import freechips.rocketchip.regmapper.{RegField, RegFieldDesc}
import freechips.rocketchip.subsystem.BaseSubsystem
import freechips.rocketchip.tilelink.{TLFragmenter, TLRegisterNode}
import freechips.rocketchip.util.{ElaborationArtefacts, ClockGate}
import org.chipsalliance.cde.config._

case class CNNParams(
  Addr: BigInt = 0x06400000L,
  dataWidth: Int = 32
)

case object CNNKeys extends Field[Option[CNNParams]](None)

class CNN()(implicit p: Parameters) extends BlackBox with HasBlackBoxResource {
  override def desiredName = "CNN_System_Top"
  val io = IO (new Bundle{
    val clk            = Input(Clock())
    val rst_n          = Input(Bool())
    val start          = Input(Bool())
    val config_mode    = Input(Bool())
    val cpu_data_in    = Input(UInt(20.W))
    val cpu_wr_en      = Input(Bool())
    val ready_for_data = Output(Bool())
    val classification = Output(UInt(3.W))
    val done_all       = Output(Bool())
  })
  addResource("/vsrc/CNN_System_Top.sv")
}

class CNNDevice(val params: CNNParams, beatBytes: Int = 8)(implicit p: Parameters)
  extends LazyModule {
  val device = new SimpleDevice("CNNmod", Seq("SoC,CNN"))
  val mmioNode = TLRegisterNode(
    address   = Seq(AddressSet(params.Addr, 4096-1)),
    device    = device,
    beatBytes = beatBytes,
    concurrency = 1
  )
  lazy val module = new CNNDeviceImp(this)
}

class CNNDeviceImp(outer: CNNDevice)(implicit p: Parameters)
  extends LazyModuleImp(outer) {

  val params = outer.params
  val cnn = Module(new CNN())

  // ==== MMIO Registers ====
  val r_reset       = RegInit(false.B)
  val r_start       = RegInit(false.B)
  val r_config_mode = RegInit(false.B)
  val r_wr_en       = RegInit(false.B)
  val r_data_in     = RegInit(0.U(20.W))

  val r_ready          = RegInit(false.B)
  val r_classification = RegInit(0.U(3.W))
  val r_done           = RegInit(false.B)

  // T?o xung WREN t? su?n lên
  val r_wr_en_dly = RegNext(r_wr_en, false.B)
  val wr_en_pulse = r_wr_en && !r_wr_en_dly

  // ==========================================================
  // CLOCK GATING: ÉP T?C Ð? RTL B?NG T?C Ð? CPU MMIO
  // ==========================================================
  val cnn_clk_en = Wire(Bool())

  when (r_start && !r_config_mode && cnn.io.ready_for_data) {
    // Trong pha bom Data, clock CH? N?Y M?T NH?P khi CPU nh?p wr_en_pulse
    cnn_clk_en := wr_en_pulse
  } .otherwise {
    // Các pha còn l?i (Config Weights, Inference, Reset) ch?y Clock liên t?c
    cnn_clk_en := true.B
  }

  // ?? S? D?NG ClockGate() T? RocketChip Ð? T?O GATED CLOCK CHU?N
  val cnn_gated_clk = ClockGate(clock, cnn_clk_en)

  // ==========================================================
  // Connect Chipyard -> CNN
  // ==========================================================
  cnn.io.clk         := cnn_gated_clk
  cnn.io.rst_n       := !r_reset
  cnn.io.start       := r_start
  cnn.io.config_mode := r_config_mode
  cnn.io.cpu_data_in := r_data_in
  cnn.io.cpu_wr_en   := wr_en_pulse

  // ==========================================================
  // CNN -> Chipyard (Ch?t k?t qu? an toàn)
  // ==========================================================
  r_ready := cnn.io.ready_for_data

  val r_start_dly = RegNext(r_start, false.B)
  val start_pulse = r_start && !r_start_dly

  when (r_reset || start_pulse) {
    r_done           := false.B
    r_classification := 0.U
  } .elsewhen (cnn.io.done_all) {
    r_done           := true.B
    r_classification := cnn.io.classification
  }

  // ==== MMIO Mapping ====
  outer.mmioNode.regmap(
    CNNRegs.control   -> Seq(
      RegField(1, r_reset,       RegFieldDesc("r_reset", "reset")),
      RegField(1, r_start,       RegFieldDesc("start","Start CNN inference")),
      RegField(1, r_config_mode, RegFieldDesc("config_mode", "CNN config mode")),
      RegField(1, r_wr_en,       RegFieldDesc("cpu_wr_en", "Enable write SRAM")),
      RegField(28)),
    CNNRegs.data_in -> Seq(
      RegField(20, r_data_in,    RegFieldDesc("data_in", "20-bit data input")),
      RegField(12)),
    CNNRegs.ready -> Seq(
      RegField.r(1, r_ready, RegFieldDesc("ready", "CNN ready", volatile = true)),
      RegField(31)),
    CNNRegs.classification -> Seq(
      RegField.r(3, r_classification, RegFieldDesc("classification", "CNN result", volatile = true)),
      RegField(29)),
    CNNRegs.done -> Seq(
      RegField.r(1, r_done, RegFieldDesc("done", "CNN completed", volatile = true)),
      RegField(31)
    )
  )
}

trait CanHavePeripheryCNNmod { this: BaseSubsystem =>
  private val portName = "cnn"
  val CNNDeviceOpt: Option[CNNDevice] = p(CNNKeys).map { params =>
    val cnnWrapper = LazyModule(new CNNDevice(params, pbus.beatBytes)(p))
    pbus.coupleTo(s"cnn_mmio_at_${params.Addr.toString(16)}") {
      cnnWrapper.mmioNode := TLFragmenter(pbus) := _
    }
    cnnWrapper
  }
}

class WithCNNmod (base: BigInt = 0x06400000L, width: Int = 32) extends Config((site, here, up) => {
  case CNNKeys => Some(CNNParams(Addr = base, dataWidth = width))
})

