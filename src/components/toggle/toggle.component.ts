import { Component } from '@angular/core';
import { NgValueAccessor } from 'src/api/forms/ng-value-accessor';
import { BaseInputComponent } from 'src/api/forms/base-input-component';

@Component({
    selector: 'toggle',
    templateUrl: './toggle.component.html',
    styleUrls: ['./toggle.component.scss'],
    providers: [
        NgValueAccessor.get(ToggleComponent)
    ]
})
export class ToggleComponent extends BaseInputComponent {
    
}