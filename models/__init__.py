# models/__init__.py
from ultralytics.nn import tasks
from .mim_modules import MIMAdapter, MIMHead

# Register custom modules into Ultralytics global map
# This is a bit of a hack, but standard for research prototyping with YOLO
if not hasattr(tasks, 'MIMAdapter'):
    setattr(tasks, 'MIMAdapter', MIMAdapter)
if not hasattr(tasks, 'MIMHead'):
    setattr(tasks, 'MIMHead', MIMHead)