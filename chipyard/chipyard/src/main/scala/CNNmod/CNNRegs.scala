package chipyard.CNNmod

object CNNRegs {
  val r_reset         = 0x00
  val config_mode     = 0x04
  val weight_valid    = 0x08
  val weight_data     = 0x0C
  val ecg_valid       = 0x10
  val ecg_data        = 0x14
  val start_infer     = 0x18
  val classification  = 0x1C
  val done            = 0x20
}
