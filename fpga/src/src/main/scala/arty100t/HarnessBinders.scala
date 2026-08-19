// See LICENSE for license details.
package chipyard.fpga.arty100t // Ð?i tên package thành arty100t

import chipyard._
import chipyard.harness._
import chipyard.iobinders.JTAGChipIO
import chisel3._
import chisel3.experimental.BaseModule
import freechips.rocketchip.devices.debug.HasPeripheryDebug
import freechips.rocketchip.tilelink.TLBundle
import freechips.rocketchip.util.HeterogeneousBag
import sifive.blocks.devices.gpio.{GPIOPortIO, HasPeripheryGPIOModuleImp}
import sifive.blocks.devices.spi.{HasPeripherySPI, SPIPortIO}
import sifive.blocks.devices.uart.{HasPeripheryUARTModuleImp, UARTPortIO}
import testchipip._

/*** UART ***/
class WithArty100TUARTHarnessBinder extends OverrideHarnessBinder({
  (system: HasPeripheryUARTModuleImp, th: BaseModule, ports: Seq[UARTPortIO]) => {
    th match {
      // Ð?i Arty35TwoDDRHarnessImp thành Arty100TwoDDRHarnessImp
      case ath: Arty100TwoDDRHarnessImp => ath.athOuter.io_uart_bb.bundle <> ports.head
    }
  }
})

/*** SPI ***/
class WithArty100TSPISDCardHarnessBinder extends OverrideHarnessBinder({
  (system: HasPeripherySPI, th: BaseModule, ports: Seq[SPIPortIO]) => {
    th match {
      // Ð?i Arty35TwoDDRHarnessImp thành Arty100TwoDDRHarnessImp
      case ath: Arty100TwoDDRHarnessImp => ath.athOuter.io_spi_bb.bundle <> ports.head
    }
  }
})

/*** JTAG ***/
class WithArty100TJTAGHarnessBinder extends OverrideHarnessBinder({
  (system: HasPeripheryDebug, th: BaseModule, ports: Seq[Data]) => {
    ports.map {
      case jtagIO: JTAGChipIO =>
        th match {
          // Ð?i Arty35TwoDDRHarnessImp thành Arty100TwoDDRHarnessImp
          case ath: Arty100TwoDDRHarnessImp => {
            val jtagModule = ath.athOuter.jtagModule
            jtagModule.TDO.data := jtagIO.TDO
            jtagModule.TDO.driven := true.B
            jtagIO.TCK := jtagModule.TCK
            jtagIO.TMS := jtagModule.TMS
            jtagIO.TDI := jtagModule.TDI
          }
        }
    }
  }
})

/*** GPIO ***/
class WithArty100TGPIOHarnessBinder extends OverrideHarnessBinder({
  (system: HasPeripheryGPIOModuleImp, th: BaseModule, ports: Seq[GPIOPortIO]) => {
    th match {
      // Ð?i Arty35TwoDDRHarnessImp thành Arty100TwoDDRHarnessImp
      case th: Arty100TwoDDRHarnessImp => {
        (th.athOuter.io_gpio_bb zip ports).map{ case (gpio, port) =>
          gpio.bundle <> port
        }
      }
    }
  }
})

/*** Tie off TSI ***/
class WithArty100TTSITieoff extends OverrideHarnessBinder ({
  (system: CanHavePeripheryTLSerial, th: BaseModule, ports: Seq[ClockedIO[SerialIO]]) => {
    th match {
      // Ð?i Arty35TwoDDRHarnessImp thành Arty100TwoDDRHarnessImp
      case athOuter: Arty100TwoDDRHarnessImp => {
        ports.map({ port =>
          val bits = port.bits
          port.clock := athOuter.harnessBinderClock
          val ram = TSIHarness.connectRAM(system.serdesser.get, bits, athOuter.harnessBinderReset)
          TSIHarness.tieoff(ram.module.io.tsi)
        })
      }
    }
  }
})
