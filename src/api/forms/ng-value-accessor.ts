import { NG_VALUE_ACCESSOR } from '@angular/forms';
import { forwardRef } from '@angular/core';

export class NgValueAccessor {
    static get<T>(component: T) {
        return {
            provide: NG_VALUE_ACCESSOR,
            useExisting: forwardRef(() => component),
            multi: true
        }
    }
}