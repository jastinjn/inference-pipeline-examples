import os
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst
import sys
# append the dir containing helper functions from deepstream-python-apps
sys.path.append('/opt/nvidia/deepstream/deepstream-6.3/sources/deepstream_python_apps/apps')
from common.bus_call import bus_call
from common.FPS import PERF_DATA
import pyds
import numpy as np

perf_data = None
perf_hist = []
fps = None
frame_counter = 0

def osd_sink_pad_buffer_probe(pad,info,u_data):
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list

    while l_frame:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break
        
        # Update frame rate through this probe
        stream_index = "stream{0}".format(frame_meta.pad_index)
        global perf_data
        perf_data.update_fps(stream_index)
        
        global frame_counter
        global fps
        frame_counter += 1
        if frame_counter % 30 == 0:
            fps_now = perf_data.all_stream_fps[stream_index].get_fps()
            perf_hist.append(fps_now)
            fps = fps_now
	    
        display_meta=pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        display_meta.num_labels = 1
        py_nvosd_text_params = display_meta.text_params[0]
        # Setting display text to be shown on screen
        # Note that the pyds module allocates a buffer for the string, and the
        # memory will not be claimed by the garbage collector.
        # Reading the display_text field here will return the C address of the
        # allocated string. Use pyds.get_string() to get the string content.
        py_nvosd_text_params.display_text = f"THROUGHPUT: {fps}FPS"

        py_nvosd_text_params.x_offset = 50
        py_nvosd_text_params.y_offset = 400
        py_nvosd_text_params.font_params.font_name = "Arial"
        py_nvosd_text_params.font_params.font_size = 10
        py_nvosd_text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)

        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
        
        try:
            l_frame=l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK

def cb_newpad(decodebin, decoder_src_pad,data):
    print("In cb_newpad\n")
    caps=decoder_src_pad.get_current_caps()
    if not caps:
        caps = decoder_src_pad.query_caps()
    gststruct=caps.get_structure(0)
    gstname=gststruct.get_name()
    source_bin=data
    features=caps.get_features(0)

    # Need to check if the pad created by the decodebin is for video and not
    # audio.
    print("gstname=",gstname)
    if(gstname.find("video")!=-1):
        # Link the decodebin pad only if decodebin has picked nvidia
        # decoder plugin nvdec_*. We do this by checking if the pad caps contain
        # NVMM memory features.
        print("features=",features)
        if features.contains("memory:NVMM"):
            # Get the source bin ghost pad
            bin_ghost_pad=source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                sys.stderr.write("Failed to link decoder src pad to source bin ghost pad\n")
        else:
            sys.stderr.write(" Error: Decodebin did not pick nvidia decoder plugin.\n")

def decodebin_child_added(child_proxy,Object,name,user_data):
    print("Decodebin child added:", name, "\n")
    if(name.find("decodebin") != -1):
        Object.connect("child-added",decodebin_child_added,user_data)

    if "source" in name:
        source_element = child_proxy.get_by_name("source")
        if source_element.find_property('drop-on-latency') != None:
            Object.set_property("drop-on-latency", True)

def create_source_bin(index,uri):
    print("Creating source bin")

    # Create a source GstBin to abstract this bin's content from the rest of the
    # pipeline
    bin_name="source-bin-%02d" %index
    print(bin_name)
    nbin=Gst.Bin.new(bin_name)
    if not nbin:
        sys.stderr.write(" Unable to create source bin \n")

    # Source element for reading from the uri.
    # We will use decodebin and let it figure out the container format of the
    # stream and the codec and plug the appropriate demux and decode plugins.
    
    uri_decode_bin=Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
    if not uri_decode_bin:
        sys.stderr.write(" Unable to create uri decode bin \n")
    # We set the input uri to the source element
    uri_decode_bin.set_property("uri",uri)
    # Connect to the "pad-added" signal of the decodebin which generates a
    # callback once a new pad for raw data has beed created by the decodebin
    uri_decode_bin.connect("pad-added",cb_newpad,nbin)
    uri_decode_bin.connect("child-added",decodebin_child_added,nbin)

    # We need to create a ghost pad for the source bin which will act as a proxy
    # for the video decoder src pad. The ghost pad will not have a target right
    # now. Once the decode bin creates the video decoder and generates the
    # cb_newpad callback, we will set the ghost pad target to the video decoder
    # src pad.
    Gst.Bin.add(nbin,uri_decode_bin)
    bin_pad=nbin.add_pad(Gst.GhostPad.new_no_target("src",Gst.PadDirection.SRC))
    if not bin_pad:
        sys.stderr.write(" Failed to add ghost pad in source bin \n")
        return None
    return nbin

# DeepStream configuration for the application

def main():

    # initialise performance tracking
    global perf_data
    perf_data = PERF_DATA(1)
    
    # Initialize GST
    Gst.init(None)
   
    # Create the pipeline
    pipeline = Gst.Pipeline()

    # Create the video source
    uri_name = 'file:///opt/nvidia/deepstream/deepstream-6.3/samples/streams/sample_720p.mp4'
    source_bin=create_source_bin(0, uri_name)
    if not source_bin:
        sys.stderr.write("Unable to create source bin \n")
    pipeline.add(source_bin)
    
    # Streammux
    streammux = Gst.ElementFactory.make('nvstreammux', 'streammux')
    if not streammux:
        print("Streammux element could not be created!")
        sys.exit(1)
    pipeline.add(streammux)

    # Set streammux properties
    streammux.set_property('batch-size', 1)
    streammux.set_property('width', 852)
    streammux.set_property('height', 480)
    
    sinkpad= streammux.get_request_pad("sink_0")
    if not sinkpad:
        sys.stderr.write("Unable to create sink pad bin \n")
    	
    srcpad=source_bin.get_static_pad("src")
    if not srcpad:
        sys.stderr.write("Unable to create src pad bin \n")
    
    srcpad.link(sinkpad) 

    # Create the primary inference element (GIE)
    primary_gie = Gst.ElementFactory.make('nvinfer', 'primary-gie')
    if not primary_gie:
        print("Primary GIE element could not be created!")
        sys.exit(1)

    primary_gie.set_property('config-file-path', '/home/siot/DeepStream-Yolo/config_infer_primary_yoloV8.txt')
      
    # Create nvvidconv (video converter)
    nvvidconv = Gst.ElementFactory.make('nvvideoconvert', 'nvvidconv')
    if not nvvidconv:
        print("Nvvidconv element could not be created!")
        sys.exit(1)

    # Create the OSD (On-Screen Display)
    osd = Gst.ElementFactory.make('nvdsosd', 'osd')
    if not osd:
        print("OSD element could not be created!")
        sys.exit(1)

    osd.set_property('gpu-id', 0)
    osd.set_property('display-clock', 0)
    osd.set_property('x-clock-offset', 800)
    osd.set_property('y-clock-offset', 820)
    osd.set_property('clock-font-size', 12)
    osd.set_property('clock-color', 0xff0000ff)
    
    # Create the sink (display output)
    sink = Gst.ElementFactory.make('nv3dsink', 'sink')
    if not sink:
        print("Sink element could not be created!")
        sys.exit(1)
    # if live video source or max 30 fps framerate, set to true
    sink.set_property('sync', 0)

    # Add elements to pipeline
    pipeline.add(primary_gie)
    pipeline.add(nvvidconv)
    pipeline.add(osd)
    pipeline.add(sink)

    # Create queue elements for asynchronous pipeline
    queue1=Gst.ElementFactory.make("queue","queue1")
    queue2=Gst.ElementFactory.make("queue","queue2")
    queue3=Gst.ElementFactory.make("queue","queue3")
    pipeline.add(queue1)
    pipeline.add(queue2)
    pipeline.add(queue3)

    # Link elements
    streammux.link(queue1)
    queue1.link(primary_gie)
    primary_gie.link(queue2)
    queue2.link(nvvidconv)
    nvvidconv.link(queue3)
    queue3.link(osd)
    osd.link(sink)
    
    # Initialize loop
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect ("message", bus_call, loop)
    
    # Add probes
    
    osdsinkpad = osd.get_static_pad("sink")
    if not osdsinkpad:
        sys.stderr.write(" Unable to get sink pad of nvosd \n")
    else:
        osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)
        
    # Start playing the pipeline
    pipeline.set_state(Gst.State.PLAYING)

    # Start the main loop
    try:  
        loop.run()
    except Exception as e:
        print("Error during pipeline execution:", e)
        pass
   
    pipeline.set_state(Gst.State.NULL)
    
    avg_fps = np.mean(np.array(perf_hist))
    print(f"MEAN_FPS: {avg_fps:.2f}")
   

if __name__ == "__main__":
    sys.exit(main())
