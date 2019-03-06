#!/bin/sh
# Tcl ignores the next line -*- tcl -*- \
exec wish "$0" -- "$@"



set all_video_frames {"frame1.ppm" "frame2.ppm" "frame3.ppm"}
set current_file_name "frame1.ppm"
set current_file_idx 0 

set output_file 0

proc load_all_files { filename } {
    global all_video_frames
    global current_file_name current_file_idx

    set all_video_frames {}

    set fid [open $filename r]
    foreach f [split [read $fid] \n] {
        lappend all_video_frames [split $f ,]
    }
    close $fid

    set current_file_idx 0
    set current_file_name [lindex [lindex $all_video_frames $current_file_idx] 0]
}

# Draw a circle around.
proc draw_circle {c xi yi} {
    set r 7
    set x0 [expr $xi - $r]
    set x1 [expr $xi + $r]
    set y0 [expr $yi - $r]
    set y1 [expr $yi + $r]

    $c create oval $x0 $y0 $x1 $y1 -outline red -width 2 -tags circle
}

proc next_image {} {

    global video_frame current_file_name current_file_idx all_video_frames
    global video_frame
    
    # Save all the key points that have been marked.

    # Display the id of marked key points and clear the video frame canvas.

    .images.video delete circle

    # Load the next frame.
    incr current_file_idx
    if {$current_file_idx < [llength $all_video_frames] } {
        set current_entry [lindex $all_video_frames $current_file_idx]
        set current_file_name [lindex $current_entry 0]
        set video_frame [image create photo video_frame -file $current_file_name]

        for {set i 1} {$i < [expr [llength $current_entry] - 1]} {} {

            set x [lindex $current_entry $i]
            incr i
            set y [lindex $current_entry $i]
            incr i
            set l [lindex $current_entry $i]
            incr i

            set x [expr $x/4]
            set y [expr $y/4]
            puts "$x $y $l"
            draw_circle .images.video $x $y
            .images.video create text $x $y -text $l -anchor sw -tags circle
        }

        puts "Moving to $current_file_name"
    } 
}


# ==========================================
if {$argc != 1} {
    puts "Usage: gui input-file-list"
    exit
} else {
   load_all_files [lindex $argv 0] 

}

set video_frame [image create photo video_frame -file $current_file_name]

# Container of the two images.
frame .images -width 720 -height 500
# Keyboard shortcuts
bind .images <space> { next_image }
bind .images <Key-q> { exit }
focus .images

# GUI to display a video frame along side of the reference image
canvas .images.video -width [image width $video_frame] -height [image height $video_frame] 
.images.video create image 0 0 -image video_frame -anchor nw -tags images

pack .images -fill x
pack .images.video -side left

bind .images.video <Button-1> {next_image}

