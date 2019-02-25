pub struct Ptr(usize);

impl Ptr {
    pub fn from_addr(borrow_addr: &f64) -> Ptr {
        Ptr((borrow_addr as *const f64) as usize)
    }

    pub fn from_ptr(ptr: *const f64) -> Ptr {
        Ptr(ptr as usize)
    }

    pub fn as_ptr(&self) -> *const f64 {
        self.0 as *const f64
    }

    pub fn as_mut_ptr(&self) -> *mut f64 {
        self.0 as *mut f64
    }
}
