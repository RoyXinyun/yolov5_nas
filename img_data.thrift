namespace py img_data

struct Data {
    1: string img
}

struct Result {
    1: string result
}

service format_data {
    Data do_format(1:Data data),
}