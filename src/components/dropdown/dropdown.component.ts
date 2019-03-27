import { Component, Input } from '@angular/core';
import { DropdownOption } from 'src/api/dropdown/dropdown-option';
import { DropdownOptionGroup } from 'src/api/dropdown/dropdown-option-group';
import { BaseInputComponent } from 'src/api/forms/base-input-component';

@Component({
    selector: 'dropdown',
    templateUrl: './dropdown.component.html',
    styleUrls: [
        './dropdown.component.scss'
    ]
})
export class DropdownComponent extends BaseInputComponent {
    @Input() options: DropdownOption[] | DropdownOptionGroup[];
    @Input() grouped: boolean;
    @Input() disabled: boolean;

    
}