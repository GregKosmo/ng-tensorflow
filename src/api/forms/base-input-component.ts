import { ControlValueAccessor } from '@angular/forms';

export class BaseInputComponent implements ControlValueAccessor {
    value: any;
    onChange = (value: any) => {};
    onTouch = () => {};

    writeValue(value: any) {
        this.value = value;
    }

    registerOnChange(callback: (value: any) => void) {
        this.onChange = callback;
    }

    registerOnTouched(callback: () => void) {
        this.onTouch = callback;
    }
}