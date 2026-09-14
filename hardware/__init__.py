"""Hardware Layer.

Thin wrappers around physical (or simulated) sensors/actuators. Every
component here implements a small interface so higher layers never import
cv2/RPi.GPIO/serial directly — they depend on the interface, and hardware
absence is handled here, not upstream.
"""
