namespace py feature_data

struct Data {
    1: string outs
    2: string shapes
    3: string feats_shapes
    4: string origin_shape
    5: string resize_shape
    6: string inputs_mean
    7: string inputs_std
}

struct Result {
    1: string det_string
    2: string time_string
    3: i32 length
}

service format_data {
    Result do_format(1:Data data),
}