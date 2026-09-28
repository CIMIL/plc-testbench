import numpy as np
import numpy.random as npr

from plctestbench.worker import Worker

from .settings import (
    BinomialPLSSettings,
    GilbertElliotPLSSettings,
    MetronomePLSSettings,
    CustomMaskPLSSettings,
    Settings,
)


class PacketLossSimulator(Worker):
    """Base class for packet-loss models.

    Subclasses implement
    [`tick()`][plctestbench.loss_simulator.PacketLossSimulator.tick] to decide
    whether each packet is lost. The
    [`run()`][plctestbench.loss_simulator.PacketLossSimulator.run] method
    expands that packet-level decision to every sample in the packet.
    """

    def __init__(self, settings: Settings) -> None:
        """Initialize the common packet-loss state.

        Other Parameters:
            seed (int): Seed used by simulators that generate random values.
            packet_size (int): Number of audio samples in each packet.
        """
        super().__init__(settings)
        self.packet_size = settings.get("packet_size")

    def run(self, num_samples, id) -> np.ndarray:
        """Compute the positions of lost samples in an audio track.

        Args:
            num_samples (int): Total number of samples in the audio track.
            id (str): Identifier shown by the progress monitor.

        Returns:
            A NumPy array containing the indices of all lost samples.
        """
        lost_samples_idx = []
        for idx in self.progress_monitor(range(num_samples), desc=f"{str(self)}|{id}"):
            if (idx % self.packet_size) == 0:
                lost_packet = self.tick()
            if lost_packet:
                lost_samples_idx.append(idx)

        return np.array(lost_samples_idx)

    def __str__(self) -> str:
        return self.__class__.__name__ + "_s" + str(self.settings.get("seed"))

    def tick(self) -> bool:
        """Decide whether the current packet is lost.

        Raises:
            NotImplementedError: Always. Subclasses must implement this
                method.
        """
        raise NotImplementedError


class BinomialPLS(PacketLossSimulator):
    """
    Implements a binomial distribution (Bernoulli trials)
    to be used as a loss model in packet/sample loss simulators.
    """

    def __init__(self, settings: BinomialPLSSettings) -> None:
        """Initialize a binomial packet-loss simulator.

        Parameters:
            seed (int): Seed for the random number generator. Defaults to
                ``1``.
            packet_size (int): Number of audio samples in each packet.
                Defaults to ``32``.
            per (float): Packet error ratio, expressed as a probability in the
                range ``[0, 1]``. Defaults to ``0.0001``.
        """
        super().__init__(settings)
        self.per = settings.get("per")
        npr.seed(self.settings.get("seed"))

    def tick(self) -> bool:
        """Perform one Bernoulli trial.

        Returns:
            ``True`` if the current packet is lost; otherwise ``False``.
        """
        b_trial_result = npr.random() <= self.per
        return b_trial_result


class MetronomePLS(PacketLossSimulator):
    """Generate deterministic, periodic bursts of packet loss.

    Each cycle contains ``period`` packets. The first ``duration`` packets are
    lost and the remainder are received. A positive ``offset`` delays the
    first burst. The simulator advances the cycle one packet at a time through
    [`tick()`][plctestbench.loss_simulator.MetronomePLS.tick].

    Examples:
        With ``period=10``, ``duration=3``, and ``offset=0``::

            Packet: 0 1 2 3 4 5 6 7 8 9 | 10 11 12 ...
            State:  L L L R R R R R R R | L  L  L  ...

        With ``period=10``, ``duration=3``, and ``offset=4``::

            Packet: 0 1 2 3 4 5 6 7 8 9 | 10 11 ...
            State:  R R R R L L L R R R | R  L  ...
    """

    def __init__(self, settings: MetronomePLSSettings) -> None:
        """Initialize a metronome packet-loss simulator.

        Parameters:
            seed (int): Seed reserved for consistency with other packet-loss
                simulators. Defaults to ``1``.
            packet_size (int): Number of audio samples in each packet.
                Defaults to ``32``.
            period (int): Length of a complete loss/no-loss cycle in packets.
                Defaults to ``100``.
            duration (int): Number of consecutive lost packets in each cycle.
                Defaults to ``5``.
            offset (int): Number of packets by which to delay the first loss
                burst. Defaults to ``0``.
        """
        super().__init__(settings)
        self.period = settings.get("period")
        self.duration = settings.get("duration")
        self.offset = settings.get("offset")
        self.counter = -self.offset

    def tick(self) -> bool:
        """Advance the cycle and determine whether the current packet is lost.

        Returns:
            ``True`` if the current packet is lost; otherwise ``False``.
        """
        self.counter += 1
        if self.counter == self.period:
            self.counter = 0
        if self.counter < self.duration and self.counter >= 0:
            return True
        return False


class GilbertElliotPLS(PacketLossSimulator):
    """Simulate packet loss with the two-state Gilbert--Elliott model.

    The implementation is adapted from the MIT-licensed
    [sim2net implementation](https://github.com/mkalewski/sim2net/blob/master/sim2net/packet_loss/gilbert_elliott.py).
    """

    def __init__(self, settings: GilbertElliotPLSSettings) -> None:
        """Initialize a Gilbert--Elliott packet-loss simulator.

        Parameters:
            seed (int): Seed for the random number generator. Defaults to
                ``1``.
            packet_size (int): Number of audio samples in each packet.
                Defaults to ``32``.
            p (float): Probability of transitioning from the good state to the
                bad state. Defaults to ``0.001``.
            r (float): Probability of transitioning from the bad state to the
                good state. Defaults to ``0.05``.
            h (float): Probability of receiving a packet in the bad state.
                Defaults to ``0.5``.
            k (float): Probability of receiving a packet in the good state.
                Defaults to ``0.999999``.
        """
        super().__init__(settings)
        npr.seed(self.settings.get("seed"))
        p = settings.get("p")
        r = settings.get("r")
        h = settings.get("h")
        k = settings.get("k")

        b = 1.0 - h
        g = 1.0 - k
        # ( current state: 'G' or 'B',
        #   transition probability,
        #   current packet error rate )
        self.state_g = ("G", p, g)
        self.state_b = ("B", r, b)
        self.current_state = self.state_g

    def tick(self) -> bool:
        """Advance the model and determine whether the current packet is lost.

        Returns:
            ``True`` if the current packet is lost; otherwise ``False``.
        """
        transition = npr.random()
        if transition <= self.current_state[1]:
            if self.current_state[0] == "G":
                self.current_state = self.state_b
            else:
                self.current_state = self.state_g
        loss = npr.random()
        if loss <= self.current_state[2]:
            return True
        return False


class CustomMaskPLS(PacketLossSimulator):
    """Apply a predefined binary packet-loss mask.

    Each character controls one packet: ``1`` marks a lost packet and ``0``
    marks a received packet. Setting ``invert`` reverses those meanings. Once
    the mask is exhausted, all subsequent packets are received.

    The mask length should normally equal ``num_samples / packet_size``.

    Attributes:
        mask: Binary string defining the packet-loss pattern.
        invert: Whether to invert the meaning of the mask values.
        counter: Current position in the mask.
    """

    def __init__(self, settings: CustomMaskPLSSettings):
        """Initialize a custom-mask packet-loss simulator.

        Parameters:
            seed (int): Seed reserved for consistency with other packet-loss
                simulators. Defaults to ``1``.
            packet_size (int): Number of audio samples in each packet.
                Defaults to ``32``.
            mask (str): String whose characters define the packet-loss pattern.
                Defaults to ``"0"``.
            invert (bool): Whether ``0`` represents loss and ``1`` represents
                reception. Defaults to ``False``.
        """
        super().__init__(settings)
        self.mask: str = settings.get("mask")
        self.invert: bool = settings.get("invert")
        self.counter: int = 0

    def tick(self):
        """Consume one mask value and determine whether the packet is lost.

        Result:
            ``True`` if the current packet is lost; otherwise ``False``.
        """
        try:
            lost = bool(int(self.mask[self.counter]))
            lost = not lost if self.invert else lost
        except IndexError:
            lost = False
        except ValueError:
            lost = False
        self.counter += 1
        return lost
