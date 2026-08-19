// See LICENSE for license details.
package chipyard.fpga.arty100t // 1. Ð?i tên package thành arty100t

import chipyard._
import chipyard.harness._
import chisel3._
import chisel3.util._
import freechips.rocketchip.diplomacy._
import freechips.rocketchip.subsystem.SystemBusKey
import freechips.rocketchip.tilelink._
import org.chipsalliance.cde.config.Parameters
import sifive.blocks.devices.gpio.{GPIOPortIO, PeripheryGPIOKey}
import sifive.blocks.devices.spi.{PeripherySPIKey, SPIPortIO}
import sifive.blocks.devices.uart._
import sifive.fpgashells.clocks.{ClockGroup, ClockSinkNode, PLLFactoryKey, ResetWrangler}
import sifive.fpgashells.ip.xilinx.IBUF
import sifive.fpgashells.shell._

// 2. Ð?i tên Class Harness và Trait CustomOverlays sang Arty100T
class Arty100TwoDDRHarness(override implicit val p: Parameters) extends Arty100TShellCustomOverlays
{
  def dp = designParameters

  // ========= Clock =================
  val sysClkNode = dp(ClockInputOverlayKey).head.place(ClockInputDesignInput()).overlayOutput.node
  val harnessSysPLL = dp(PLLFactoryKey)()
  harnessSysPLL := sysClkNode

  // create and connect to the dutClock
  val dutFreqMHz = (dp(SystemBusKey).dtsFrequency.get / (1000 * 1000)).toInt
  // C?p nh?t câu l?nh in log ra màn hình console khi build sang Arty100T
  println(s"Arty100T FPGA Base Clock Freq: ${dutFreqMHz} MHz")
  val dutWrangler = LazyModule(new ResetWrangler())
  val dutGroup = ClockGroup()
  
  // !!! ÐÃ S?A: Thêm dòng kh?i t?o Node cho dutClock ? dây !!!
  val dutClock = ClockSinkNode(freqMHz = dutFreqMHz)
  dutClock := dutWrangler.node := dutGroup := harnessSysPLL

  /*** JTAG ***/
  val jtagModule = dp(JTAGDebugOverlayKey).head.place(JTAGDebugDesignInput()).overlayOutput.jtag

  /*** UART ***/
  val io_uart_bb = BundleBridgeSource(() => new UARTPortIO(dp(PeripheryUARTKey).headOption.getOrElse(UARTParams(0))))
  val uartOverlay = dp(UARTOverlayKey).head.place(UARTDesignInput(io_uart_bb))

  /*** GPIO ***/
  val io_gpio_bb = dp(PeripheryGPIOKey).map(p => BundleBridgeSource(() => (new GPIOPortIO(p))))
  (dp(GPIOOverlayKey) zip dp(PeripheryGPIOKey)).zipWithIndex.map { case ((placer, params), i) =>
    placer.place(GPIODesignInput(params, io_gpio_bb(i)))
  }

  /*** SDIO ***/
  val io_spi_bb = BundleBridgeSource(() => (new SPIPortIO(dp(PeripherySPIKey).head)))
  dp(SPIOverlayKey).head.place(SPIDesignInput(dp(PeripherySPIKey).head, io_spi_bb))

  // Module implementation
  // 3. Kh?i t?o dúng Class Implement c?a Arty100
  override lazy val module = new Arty100TwoDDRHarnessImp(this)
}

// 4. Ð?i tên Class Implement tuong ?ng
class Arty100TwoDDRHarnessImp(_outer: Arty100TwoDDRHarness) extends LazyRawModuleImp(_outer)
  with HasHarnessInstantiators
{
  val athOuter = _outer

  val reset = IO(Input(Bool()))
  _outer.xdc.addBoardPin(reset, "reset")

  val reset_ibuf = Module(new IBUF)
  reset_ibuf.io.I := reset

  val sysclk: Clock = _outer.sysClkNode.out.head._1.clock

  _outer.pllReset := !reset_ibuf.io.O

  def referenceClockFreqMHz = _outer.dutFreqMHz
  def referenceClock = _outer.dutClock.in.head._1.clock
  def referenceReset = _outer.dutClock.in.head._1.reset
  def success = { require(false, "Unused"); false.B }

  childClock := referenceClock
  childReset := referenceReset

  instantiateChipTops()
}
