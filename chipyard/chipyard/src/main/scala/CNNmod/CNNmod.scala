package chipyard.CNNmod

import Chisel._
import chisel3.WireInit
import chisel3.util.HasBlackBoxResource
import freechips.rocketchip.diplomacy._
import freechips.rocketchip.regmapper.{RegField, RegFieldDesc}
import freechips.rocketchip.subsystem.BaseSubsystem
import freechips.rocketchip.tilelink.{TLFragmenter, TLRegisterNode}
import freechips.rocketchip.util.ElaborationArtefacts
import org.chipsalliance.cde.config._


case class CNNParams  //dinh nghia tham so (dia chi va do rong cho IP)
(
  Addr: BigInt = 0x06400000L,
  dataWidth: Int = 32
){}

case object CNNKeys extends Field[Option[CNNParams]](None)

class CNN()(implicit p: Parameters) extends BlackBox with HasBlackBoxResource {
  override def desiredName = "CNN_Wrapper"
  val io = IO (new Bundle{
    // Clock and reset
    val clk = Input(Clock())
    val rst_n = Input(Bool())

    // Weight SRAM load path
    val config_mode = Input(Bool())
    val weight_valid = Input(Bool())
    val weight_data = Input(UInt(20.W))

    // ECG sample input
    val ecg_valid = Input(Bool())
    val ecg_data = Input(UInt(8.W))

    // Control inference
    val start_infer = Input(Bool())

    // Outputs
    val classification = Output(UInt(3.W))
    val done = Output(Bool())
  })
  addResource("/vsrc/CNN_System_Top.sv")
}

class CNNDevice(val params: CNNParams, beatBytes: Int = 8)(implicit p: Parameters)  //TileLink Peripheral
  extends LazyModule {
  val device = new SimpleDevice("CNNmod", Seq("SoC,CNN"))
  val mmioNode = TLRegisterNode(
    address     = Seq(AddressSet(params.Addr, 4096-1)),
    device      = device,
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
  val r_reset        = RegInit(false.B)
  val r_config_mode  = RegInit(false.B)
  
  val r_weight_data  = RegInit(0.U(20.W))
  val r_ecg_data     = RegInit(0.U(8.W))

  // Các thanh ghi m?c luu giá tr? ghi t? MMIO
  val r_weight_valid = RegInit(false.B)
  val r_ecg_valid    = RegInit(false.B)
  val r_start_infer  = RegInit(false.B)

  // Dùng RegNext d? t?o xung edge-detection (su?n lên)
  val weight_valid_dly  = RegNext(r_weight_valid, false.B)
  val weight_valid_pulse = r_weight_valid && !weight_valid_dly

  val ecg_valid_dly     = RegNext(r_ecg_valid, false.B)
  val ecg_valid_pulse    = r_ecg_valid && !ecg_valid_dly

  val start_infer_dly   = RegNext(r_start_infer, false.B)
  val start_infer_pulse  = r_start_infer && !start_infer_dly

  val r_classification = RegInit(0.U(3.W))
  val r_done           = RegInit(false.B)

  // ==========================================================
  // Connect Chipyard -> CNN Wrapper
  // ==========================================================
  cnn.io.clk         := clock
  cnn.io.rst_n       := !r_reset

  cnn.io.config_mode  := r_config_mode
  cnn.io.weight_valid := weight_valid_pulse
  cnn.io.weight_data  := r_weight_data
  cnn.io.ecg_valid    := ecg_valid_pulse
  cnn.io.ecg_data     := r_ecg_data
  cnn.io.start_infer  := start_infer_pulse

  // ==========================================================
  // CNN Wrapper -> Chipyard
  // ==========================================================
  r_classification := cnn.io.classification
  when (r_reset || start_infer_pulse) {
    r_done             := false.B
    r_classification   := 0.U
  } .elsewhen (cnn.io.done) {
    r_done             := true.B
    r_classification   := cnn.io.classification
  }

  outer.mmioNode.regmap(
    0x00 -> Seq(RegField(1, r_reset, RegFieldDesc("r_reset", "Reset CNN wrapper"))),
    0x04 -> Seq(RegField(1, r_config_mode, RegFieldDesc("config_mode", "1 = Weight load mode, 0 = Inference mode"))),
    0x08 -> Seq(RegField(1, r_weight_valid, RegFieldDesc("weight_valid", "Pulse to write weight data"))),
    0x0C -> Seq(RegField(20, r_weight_data, RegFieldDesc("weight_data", "20-bit weight value"))),
    0x10 -> Seq(RegField(1, r_ecg_valid, RegFieldDesc("ecg_valid", "Pulse to write ECG sample"))),
    0x14 -> Seq(RegField(8, r_ecg_data, RegFieldDesc("ecg_data", "8-bit ECG sample data"))),
    0x18 -> Seq(RegField(1, r_start_infer, RegFieldDesc("start_infer", "Pulse to begin inference"))),
    0x1C -> Seq(RegField.r(3, r_classification, RegFieldDesc("classification", "CNN classification result", volatile = true))),
    0x20 -> Seq(RegField.r(1, r_done, RegFieldDesc("done", "Inference done pulse", volatile = true)))
  )
}

trait CanHavePeripheryCNNmod{ this: BaseSubsystem =>   // noi IP vao SoC
  private val portName = "cnn"

  val CNNDeviceOpt: Option[CNNDevice] = p(CNNKeys).map {params =>
    val cnnWrapper = LazyModule( new CNNDevice(params, pbus.beatBytes)(p))
    pbus.coupleTo(s"cnn_mmio_at_${params.Addr.toString(16)}") {
      cnnWrapper.mmioNode := TLFragmenter(pbus) := _
    }
    cnnWrapper
  }
}

class WithCNNmod (base: BigInt = 0x06400000L, width: Int = 32) extends Config((site, here, up) => {  //call in config
  case CNNKeys => Some(CNNParams(Addr = base, dataWidth = width))
})
